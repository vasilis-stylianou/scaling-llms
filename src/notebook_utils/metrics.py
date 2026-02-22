import math
import json
import pandas as pd
from scaling_llms.registries import GoogleDriveRunRegistry, RunManager
from scaling_llms.constants import (
    METRIC_CATS,
    METRIC_SCHEMA, 
    RUN_FILES,
    PROJECT_DEV_NAME,
    PROJECT_NAME,
)
from scaling_llms.utils.trackers import JsonlTrackerReader


def compute_expected_init_train_metrics(run: RunManager) -> dict[str, float]:
    model_config = json.loads(run.get_metadata_path(RUN_FILES.model_config).read_text())
    proba = 1 / model_config["vocab_size"]
    nll = -math.log(proba)
    ppl = math.exp(nll)
    return {"nll": nll, "ppl": ppl}

def validate_init_nll(
    run_reg: GoogleDriveRunRegistry,
    experiment_name: str, 
    metric_reader: "StepMetricsReader", # TODO
    rel_tol=1e-2
):

    for run_name in metric_reader.list_run_names():
        # Expected NLL: -log(1/vocab_size)
        run = run_reg.start_run(experiment_name, run_name, resume=True)
        init_metrics = compute_expected_init_train_metrics(run)
        expected_nll = init_metrics["nll"]

        # Observed NLL
        observed_nll = metric_reader.get_metric_value(
            run_name=run_name,
            metric_cat=METRIC_CATS.train,
            metric_name="nll",
            step=0
        )
        msg = f"Observed NLL at step=0 does not match expected value (~ -log(1/vocab_size)) for run_name='{run_name}'"
        assert math.isclose(expected_nll, observed_nll, rel_tol=rel_tol), msg

    print("Observed NLL at step=0 matches expected value (~ -log(1/vocab_size)) for all runs.")


class StepMetricsReader:
    def __init__(self, experiment_name: str, is_dev: bool = True):
        self.experiment_name = experiment_name
        self.run_reg = GoogleDriveRunRegistry(project_subdir=PROJECT_DEV_NAME if is_dev else PROJECT_NAME)
        self.run_name2jsonl_reader = self._get_run_name2jsonl_reader()

    # --- API ---
    def list_run_names(self) -> list[str]:
        return list(self.run_name2jsonl_reader.keys())

    def list_metric_names(self, run_name, metric_cat) -> list[str]:
        return (
            self.run_name2jsonl_reader[run_name]
            .get(metric_cat)
            [METRIC_SCHEMA.metric]
            .unique()
            .tolist()
        )
    
    def get_step_metric_df(
        self, 
        run_name: str, 
        metric_cat: str, 
        metric_name: str
    ) -> pd.DataFrame:
        return (
            self.run_name2jsonl_reader[run_name][metric_cat]
            .query(f"{METRIC_SCHEMA.metric} == @metric_name")
            [[METRIC_SCHEMA.step, METRIC_SCHEMA.value]]
            .sort_values(METRIC_SCHEMA.step)
            .rename(columns={METRIC_SCHEMA.value: metric_name})
            .reset_index(drop=True)
        )
    
    def get_step_metric_df_by_run(
        self, 
        metric_cat: str, 
        metric_name: str,
        run_names: list[str] | None = None,
    ) -> pd.DataFrame:
        metric_dfs = []
        run_names = run_names or self.list_run_names()
        for run_name in run_names:
            df_metric = self.get_step_metric_df(run_name, metric_cat, metric_name)
            df_metric["run_name"] = run_name
            metric_dfs.append(df_metric)
        return pd.concat(metric_dfs, ignore_index=True)
    
    def get_metric_value(
        self, 
        run_name: str, 
        metric_cat: str,
        metric_name: str, 
        step: int
    ) -> float:
        qry = f"({METRIC_SCHEMA.metric} == @metric_name) "
        qry += f"& ({METRIC_SCHEMA.step} == @step)"
        return (
            self.run_name2jsonl_reader[run_name][metric_cat]
            .query(qry)
            [METRIC_SCHEMA.value]
            .iloc[0]
        )
    

    def get_metrics_df(
        self,
        run_name: str,
        metric_cat1: str,
        metric_name1: str, 
        metric_cat2: str, 
        metric_name2: str,
        join_type="inner"
    ) -> pd.DataFrame:
        df1 = self.get_step_metric_df(run_name, metric_cat1, metric_name1)
        df2 = self.get_step_metric_df(run_name, metric_cat2, metric_name2)
        suffixes = (
            (f"_{metric_cat1}", f"_{metric_cat2}") 
            if metric_name1 == metric_name2
            else ("", "")
        )

        return pd.merge(df1, df2, on=METRIC_SCHEMA.step, how=join_type, suffixes=suffixes)



    def get_metrics_df_by_run(
        self, 
        metric_cat1: str,
        metric_name1: str, 
        metric_cat2: str, 
        metric_name2: str,
        join_type="inner",
        run_names: list[str] | None = None,
    ) -> pd.DataFrame:
        metric_dfs = []
        run_names = run_names or self.list_run_names()
        for run_name in run_names:
            df_metrics = self.get_metrics_df(
                run_name, 
                metric_cat1, 
                metric_name1, 
                metric_cat2, 
                metric_name2,
                join_type
            )
            df_metrics["run_name"] = run_name
            metric_dfs.append(df_metrics)
        return pd.concat(metric_dfs, ignore_index=True)


    # --- Internal methods ---
    def _get_run_name2jsonl_reader(self):
        df_runs = self.run_reg.get_runs_as_df(experiment_name=self.experiment_name)
        run_name2jsonl_reader = dict()
        for row in df_runs.itertuples(index=False):
            tmp_run = self.run_reg.start_run(self.experiment_name, row.run_name, resume=True)
            run_name2jsonl_reader[row.run_name] = JsonlTrackerReader(tmp_run.get_metrics_dir())
        return run_name2jsonl_reader
    
    
