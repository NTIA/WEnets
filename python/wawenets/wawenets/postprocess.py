from pathlib import Path
from typing import List

import pandas as pd


class PostProcessor:
    """
    organizes, averages, and prints or exports results from WAWEnet predictions
    """

    line_format = (
        "{wavfile} {segment_number} {channel} {sample_rate} {start_time} {stop_time} "
        "{active_level} {speech_activity} {level_normalization} {segment_step_size} "
        "{WAWEnet_mode} {model_prediction}"
    )

    def __init__(self, results: List[dict]) -> None:
        """
        creates a dataframe from results created in `wawenets.py`

        Parameters
        ----------
        results : List[dict]
            WAWEnets results as packaged in `wawenets.py`
        """
        self.results = results
        self.packaged = self._package_results()
        self.df = self._convert_to_df()

    def _package_results(self) -> List[dict]:
        """
        reformats results into a format suitable for making dataframes or
        printing lines

        Returns
        -------
        List[dict]
            WAWEnet results as packaged in `wawenets`.py
        """
        results_per_segment = list()
        for result in self.results:
            # each `result` corresponds to WAWEnet outputs for a single file.
            # a single file can have many outputs depending on the length of
            # the input file and the stride that was specified.
            #
            # here we associate within-file measurements and metadata with
            # their respective segments.
            start_stop_times = result.pop("start_stop_times")
            active_levels = result.pop("active_levels")
            speech_activities = result.pop("speech_activities")
            model_predictions = result.pop("model_prediction")
            # now what's left in `result` is metadata that's valid for the
            # entire file
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
                # copy the whole-file metadata and...
                sub_result = result.copy()
                # add segment-specific metadata to it and...
                sub_result.update(
                    segment_number=segment_number,
                    start_time=start_time,
                    stop_time=stop_time,
                    active_level=active_level,
                    speech_activity=speech_activity,
                    model_prediction=model_prediction,
                )
                # add it to our list of per-segment results
                results_per_segment.append(sub_result)
        return results_per_segment

    def _explode_column(
        self, row: pd.Series, column_name: str = None, sorted_keys: List[str] = None
    ) -> pd.Series:
        """
        extracts information from a dictionary in a cell into a series

        Parameters
        ----------
        row : pd.Series
            a row in a dataframe that contains `column_name`
        column_name : str, optional
            the name of the column to be exploded, by default None
        sorted_keys : List[str], optional
            the keys to extract from the dictionary found in `column_name`,
            in the order they should be returned, by default None

        Returns
        -------
        pd.Series
            the values in the dictionary found in `column_name` provided in the
            same order as `sorted_keys`
        """
        return pd.Series([row[column_name][item] for item in sorted_keys])

    def _exploder(
        self, df: pd.DataFrame, col_name: str, prefix: str = None
    ) -> pd.DataFrame:
        """
        given a dataframe and a column name that contains a dictionary, this
        function extracts the keys and values from that dictionary and inserts
        them as new columns into the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            a dataframe with a column that contains dictionaries
        col_name : str
            the name of the column that contains dictionaries
        prefix : str, optional
            a string to prepend to the keys of the dictionary, by default None

        Returns
        -------
        pd.DataFrame
            the input dataframe that now contains new columns that were
            extracted from a dictionary
        """
        if prefix is None:
            prefix = f"{col_name}_"
        # grab the dictionary key names from the dictionary in the first row
        sorted_keys = sorted(list(df.iloc[0][col_name].keys()))
        # add a prefix to the new column name if required
        new_col_names = [f"{prefix}{item}" for item in sorted_keys]
        # actually do the exploding
        df[new_col_names] = df.apply(
            self._explode_column, axis=1, column_name=col_name, sorted_keys=sorted_keys
        )
        return df

    def _convert_to_df(self) -> pd.DataFrame:
        """
        converts repackaged result data into a dataframe and calculates
        per-file averages using only segments with SAF > 45

        Returns
        -------
        pd.DataFrame
            a dataframe suitable for export to CSV or printing to the terminal
        """
        # make our lives easier and create a dataframe
        df = pd.DataFrame(self.packaged)

        # get the names of the predictors in these results—we use these to loop
        # through when averaging results
        predictor_names = df.iloc[0]["model_prediction"].keys()

        # explode our model predictions out into their own columns
        df = self._exploder(df, "model_prediction", "")

        # calculate per-file scores based on segments with speech activity > 45
        new_rows = list()
        # calculate per-file averages
        for wavfile, file_df in df.groupby("wavfile"):
            # just grab some info about the current file
            first_row = file_df.iloc[0]
            # extract information from segments with the acceptable level
            # of speech activity
            active_df = file_df[file_df["speech_activity"] > 45]
            # make a new row for this file with the mean values
            new_row = dict()
            # grab the mean values
            for predictor_name in predictor_names:
                new_row[predictor_name] = active_df[predictor_name].mean()
            # populate the other columns
            new_row.update(
                wavfile=first_row["wavfile"],
                segment_number="all with SAF > 45",
                channel=first_row["channel"],
                sample_rate=first_row["sample_rate"],
                duration=first_row["duration"],
                level_normalization=first_row["level_normalization"],
                segment_step_size=first_row["segment_step_size"],
                WAWEnet_mode=first_row["WAWEnet_mode"],
            )
            new_rows.append(new_row)
        # make a dataframe out of our new rows and...
        new_rows = pd.DataFrame(new_rows)
        # slap it on the end of our first DF, i guess
        df = pd.concat([df, new_rows], ignore_index=True)
        # drop the column that contained a dictionary, since we exploded
        # it
        df = df.drop(columns=["model_prediction"])
        return df

    def export_results(self, out_file: Path = None):
        """
        either prints or writes to a file the results of processing.

        Parameters
        ----------
        out_file : Path, optional
            the file path where the dataframe should be written as a CSV, if
            specified. if not, just print the entire dataframe to the terminal.
            default is None, and therefore prints to the terminal.
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
