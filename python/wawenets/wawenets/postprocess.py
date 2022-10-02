from pathlib import Path
from typing import List

import pandas as pd


class PostProcessor:
    """
    organizes, averages, and prints or expotrs results from WAWEnet predictions
    """

    line_format = (
        "{wavfile} {segment_number} {channel} {sample_rate} {start_time} {stop_time} "
        "{active_level} {speech_activity} {level_normalization} {segment_step_size} "
        "{WAWEnet_mode} {model_prediction}"
    )

    def __init__(self, results: List[dict]) -> None:
        self.results = results
        self.packaged = self._package_results()
        self.df = self._convert_to_df()

    def _package_results(self) -> List[dict]:
        """reformats results into a format suitable for making dataframes or
        printing lines"""
        results_per_segment = list()
        for result in self.results:
            start_stop_times = result.pop("start_stop_times")
            active_levels = result.pop("active_levels")
            speech_activities = result.pop("speech_activities")
            model_predictions = result.pop("model_prediction")
            per_seg_meta = zip(
                start_stop_times, active_levels, speech_activities, model_predictions
            )
            # loop over segments
            for (
                (start_time, stop_time, segment_number),
                active_level,
                speech_activity,
                model_prediction,
            ) in per_seg_meta:
                sub_result = result.copy()
                sub_result.update(
                    segment_number=segment_number,
                    start_time=start_time,
                    stop_time=stop_time,
                    active_level=active_level,
                    speech_activity=speech_activity,
                    model_prediction=model_prediction,
                )
                results_per_segment.append(sub_result)
        return results_per_segment

    def export_results(self, out_file: Path = None):
        """
        either prints or writes to a file the results of processing.
        """

        if out_file:
            self.df.to_csv(out_file)
        else:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                None,
                "display.precision",
                3,
            ):
                print(self.df)

    def _explode_column(self, row, column_name=None, sorted_keys=None):
        """extracts information from a dictionary in a cell into a series"""
        return pd.Series([row[column_name][item] for item in sorted_keys])

    def _exploder(self, df, col_name, prefix=None):
        """extracts information and keys from a dictionary found in a specific column
        and puts them into into the DF with the extracted key names"""
        if prefix is None:
            prefix = f"{col_name}_"
        sorted_keys = sorted(list(df.iloc[0][col_name].keys()))
        new_col_names = [f"{prefix}{item}" for item in sorted_keys]
        df[new_col_names] = df.apply(
            self._explode_column, axis=1, column_name=col_name, sorted_keys=sorted_keys
        )
        return df

    def _convert_to_df(self) -> pd.DataFrame:
        """converts repackaged result data into a dataframe and calculates
        per-file averages using only segments with SAF > 0.5"""
        df = pd.DataFrame(self.packaged)

        # get the names of the predictors in these results
        predictor_names = df.iloc[0]["model_prediction"].keys()

        # explode our model predictions out into their own columns
        df = self._exploder(df, "model_prediction", "")

        # calculate per-file scores based on segments with speech activity > 0.5
        new_rows = list()
        for wavfile, file_df in df.groupby("wavfile"):
            first_row = file_df.iloc[0]
            active_df = file_df[file_df["speech_activity"] > 0.5]
            # make a new row for this file with the mean values
            new_row = dict()
            # grab the mean values
            for predictor_name in predictor_names:
                new_row[predictor_name] = active_df[predictor_name].mean()
            # populate the other columns
            new_row.update(
                wavfile=first_row["wavfile"],
                segment_number="all with SAF > 0.5",
                channel=first_row["channel"],
                sample_rate=first_row["sample_rate"],
                duration=first_row["duration"],
                level_normalization=first_row["level_normalization"],
                segment_step_size=first_row["segment_step_size"],
                WAWEnet_mode=first_row["WAWEnet_mode"],
            )
            new_rows.append(new_row)
        new_rows = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows])
        df = df.drop(columns=["model_prediction"])
        return df
