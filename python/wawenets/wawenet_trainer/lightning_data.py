from typing import Any, Dict, Callable, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from pytorch_lightning import LightningDataModule

# set up lightning data module here
# set up a parent class, and then make children for the tub/IU datasets


class DFDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, metric: List[str] = None, transform=None
    ) -> None:
        super().__init__()
        self.df = df
        self.metric = metric
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Dict[str, Any]:
        """to be overloaded, probably"""
        row = self.df.iloc[index]
        sample_path = self._get_sample_filepath(row)
        sample_rate, sample = wavfile.read(sample_path)

        sample = {
            "sample_rate": sample_rate,
            # get the dimension and the type correct
            "sample_data": np.array([sample], dtype=np.float32),
            "sample_fname": sample_path.name,
        }

        if self.metric:
            sample["pred_metric"] = np.array(
                [row[metric] for metric in self.metric], dtype=np.float32
            )

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_sample_filepath(self, row):
        raise NotImplementedError()


# TODO: cleanup TUBdataset
class TUBDataset(DFDataset):
    """turn the tub DF into a dataset"""

    degraded_path = "filepath_deg"
    language = "lang"
    impairment = "con_description"

    def __init__(
        self,
        tub_df,
        root_dir,
        metric=None,
        transform=None,
        segments: Union[List[str], str] = None,
        match_segments=None,
        metadata: bool = False,
    ):
        # `metadata` specifies whether or not to send metadata
        #       along—for training we can save memory by not attaching metadata
        #       to every sample
        self.df = tub_df
        self.root_dir = Path(root_dir)
        self.metric = metric
        self.transform = transform
        self.metadata = metadata
        if isinstance(segments, list):
            self.segments = segments
        elif isinstance(segments, str):
            self.segments = [segments]
        else:
            raise TypeError("😭")
        self.match_segments = match_segments
        if match_segments:
            self.match_these = set(match_segments)
            self.dont_match = list(set(self.metric) - self.match_these)

        # torn on the wisdom of doing this bit here
        # but we gotta remove the items where there's no valid speech
        # subsegment for the subsegment specified.
        self.tub_df = tub_df
        self.index_mapper = []

        for segment_name in segments:
            valid_indices = pd.DataFrame(
                tub_df[tub_df[segment_name].notnull()].index,
                columns=["orig_index"],
            )
            valid_indices["segment"] = segment_name
            self.index_mapper.append(valid_indices)
        # have to ignore index here so we don't end up with multiple
        # items with the same index number
        self.index_mapper = pd.concat(self.index_mapper, ignore_index=True)

    def __len__(self):
        return len(self.index_mapper)

    def _get_row_seg(self, index):
        # convert from concat'ed index to original DF index
        mapped_index, current_segment = self.index_mapper.iloc[index]
        # `.iloc` doesn't work here; indices coming in will not be in the
        # form 0, 1, ..., n - 1
        row = self.tub_df.loc[mapped_index]
        return row, current_segment

    def _get_metadata(self, index):
        # this is a nice idea, but using `ConcatDataset` in our datamodule setup
        # makes using this impossible :(
        row, _ = self.index_mapper.iloc[index]
        return row

    def __getitem__(self, index):
        row, current_segment = self._get_row_seg(index)
        sample_path = self._get_sample_filepath(row)
        sample_rate, sample = wavfile.read(sample_path)

        # grab the subsegment specified
        sample = self._retrieve_subsegment(sample, sample_rate, row, current_segment)

        sample = {
            "sample_data": np.array([sample], dtype=np.float32),
        }

        if self.metric:
            if self.match_segments:
                metrics = self._select_target_segment(current_segment)
            else:
                metrics = self.metric
            sample["pred_metric"] = np.array(
                [row[metric] for metric in metrics], dtype=np.float32
            )

        if self.metadata:
            sample["sample_rate"] = sample_rate
            sample["sample_fname"] = sample_path.name
            sample["df_ind"] = index
            sample["language"] = row[self.language]
            sample["impairment"] = row[self.impairment]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _select_target_segment(self, current_segment):
        # TODO: unit test
        metrics = [
            "_".join([current_segment, item]) if item in self.match_segments else item
            for item in self.metric
        ]
        return metrics

    def _get_sample_filepath(self, row):
        return self.root_dir / row[self.degraded_path]

    def _seconds_to_samples(self, second, sample_rate):
        return np.floor(second * sample_rate)

    def _parse_subsegment(self, row, segment, sample_rate):
        subseg_name = row[segment]
        start_sec = float(subseg_name.split(" ")[1]) / 10
        stop_sec = float(subseg_name.split(" ")[2]) / 10
        start_sample = int(start_sec * sample_rate)
        stop_sample = int(stop_sec * sample_rate)
        return start_sample, stop_sample

    def _retrieve_subsegment(self, sample, sample_rate, row, subseg):
        start_sample, stop_sample = self._parse_subsegment(row, subseg, sample_rate)
        return sample[start_sample:stop_sample]


# TODO: clean up TUBDataModule:
#       decouple the dataset from the data module. dataset args should be kwargs.
#       it's not bad for now tho.
class WEnetsDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: Union[str, Path],
        batch_size: int,
        root_dir: str,
        metric: List[str],
        pt_transforms: List[Callable],
        segments: List[str],
        dataset: Dataset,
        subsample_percent: float = None,
        match_segments=None,  # TODO: relearn this
        num_workers: int = 0,
        df_preprocessor: Callable = None,
        df_preprocessor_args: Dict[str, Any] = None,
        split_column_name: str = "db",
    ):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.metric = metric
        self.transforms = pt_transforms
        self.segments = segments
        self.dataset = dataset
        self.subsample_percent = subsample_percent
        self.match_segments = match_segments
        self.num_workers = num_workers
        self.df_preprocessor = df_preprocessor
        self.df_preprocessor_args = df_preprocessor_args
        self.split_column_name = split_column_name

    def _apply_transforms(self, df: pd.DataFrame, metadata: bool = False):
        """apply transforms, return concat dataset"""
        # we have to do this because of how training is set up---
        # we've got original wavforms and IPA-ed wavforms
        # so we apply the different transforms to the dataset, resulting in multiple
        # datasets with different transforms applied
        datasets = list()
        for tf in self.transforms:
            dataset = self.dataset(
                df,
                self.root_dir,
                self.metric,
                tf,
                segments=self.segments,
                match_segments=self.match_segments,
                metadata=metadata,
            )
            datasets.append(dataset)
        return ConcatDataset(datasets)

    def setup(self, stage=None):
        """load the DF, split it, build datasets"""

        # TODO: get rid of any rows in the DF where there's somehow no
        #       valid first sample?

        # load the DF
        df = pd.read_csv(self.csv_path)

        # preprocess the DF, if requested
        if self.df_preprocessor:
            df = self.df_preprocessor(df, **self.df_preprocessor_args)

        if stage == "fit" or stage is None:
            # split by train/test/val
            train_df = df[df[self.split_column_name].str.contains("TRAIN")]
            val_df = df[df[self.split_column_name].str.contains("VAL")]

            # if we wanna train on a subset
            if self.subsample_percent:
                train_df = train_df.sample(frac=self.subsample_percent)
                val_df = val_df.sample(frac=self.subsample_percent)

            # build the datasets
            # need to build a concat dataset all relevant transforms
            self.tub_train = self._apply_transforms(train_df)
            self.tub_val = self._apply_transforms(val_df)

        if stage == "test" or stage is None:
            # same as above
            test_df = df[df[self.split_column_name].str.contains("TEST")]

            if self.subsample_percent:
                test_df = test_df.sample(frac=self.subsample_percent)
            # be sure to send the metadata through so we can do fun analysis :)
            self.tub_test = self._apply_transforms(test_df, metadata=True)

            # TODO: set up unseen dataset too! and handle one/multiple datasets gracefully

    def train_dataloader(self):
        return DataLoader(
            self.tub_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tub_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tub_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
