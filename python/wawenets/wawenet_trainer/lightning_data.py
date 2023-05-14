from typing import Any, Dict, Callable, List, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from pytorch_lightning import LightningDataModule

from NISQA.nisqa.NISQA_lib import SpeechQualityDataset

# set up datasets and the lightning data module here


class ITSDataset(Dataset):
    """a dataset suitable for loading speech segments used to train the models
    published in the 2020 ICASSP paper"""

    # the key where a relative path to a file can be found
    degraded_path_key = "filename"
    # the key that specifies the language of a speech segment
    language_key = "sourceDatasetLanguage"
    # the key where the speech processing impairment can be found
    impairment = "impairment"

    def __init__(
        self,
        its_df: pd.DataFrame,
        root_dir: str,
        metric: Union[str, list] = None,
        transform: Callable = None,
        metadata: bool = False,
        **kwargs,
    ):
        """
        initializes ITSDataset

        Parameters
        ----------
        its_df : pd.DataFrame
            a dataframe containing records in the style used to train models published
            in the 2020 ICASSP paper
        root_dir : str
            parent directory containing directories for all sub datasets (300, 301, etc.)
        metric : Union[str, list], optional
            the metric or metrics that should be used as targets, by default None
        transform : Callable, optional
            to be applied to input data or metadata, by default None
        metadata : bool, optional
            whether or not to send metadata along—for training we can save memory
            by not attaching metadata to every sample, but for testing we can do
            more sophisticated analysis by utilizing metadata, by default False
        """
        self.df = its_df
        self.root_dir = Path(root_dir)
        self.metric = metric
        self.transform = transform
        self.metadata = metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        """
        retrieves the speech segment found at `index` in `self.df`

        Parameters
        ----------
        index : int
            index in `self.df` to be loaded

        Returns
        -------
        dict
            contains `sample_data` (40k samples of speech data in a numpy array)
            at minimum, additionally `pred_metric` (a numpy array of target values)
            if requested, and optionally metadata relevant to analysis.
        """
        row = self.df.iloc[index]
        sample_path = self._get_sample_filepath(row)
        sample_rate, sample = wavfile.read(sample_path)

        sample = {
            "sample_data": np.array([sample], dtype=np.float32),
        }

        if self.metric:
            metrics = self.metric
            sample["pred_metric"] = np.array(
                [row[metric] for metric in metrics], dtype=np.float32
            )

        if self.metadata:
            sample["sample_rate"] = sample_rate
            sample["sample_fname"] = sample_path.name
            sample["df_ind"] = index
            sample["language"] = row[self.language_key]
            sample["impairment"] = row[self.impairment]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_ds_dir(self, filename: str) -> str:
        """
        extracts the subdataset directory using the known format of filenames in the
        ITS df

        Parameters
        ----------
        filename : str
            filename found using `self.degraded_path_key`

        Returns
        -------
        str
            directory where the subdataset is located
        """
        return filename.split("_")[2].split(".")[0].replace("D", "")

    def _get_sample_filepath(self, row: pd.Series) -> Path:
        """
        constructs a Path object given a row from a dataframe

        Parameters
        ----------
        row : pd.Series
            a row that's been selected from `self.df`

        Returns
        -------
        Path
            path to the file described in `row`
        """
        filename = row[self.degraded_path_key]
        ds_dir = self._get_ds_dir(filename)
        return self.root_dir / ds_dir / filename


# TODO: cleanup TUBdataset
# TODO: are we going to release our augmented TUB CSV?
class TUBDataset(Dataset):
    """a dataset suitable for loading speech segments from the TUB dataset"""

    # the key where a relative path to a file can be found
    degraded_path = "filepath_deg"
    # the key that specifies the language of a speech segment
    language = "lang"
    # the key where the speech processing impairment can be found
    impairment = "con_description"

    def __init__(
        self,
        tub_df,
        root_dir,
        metric: Union[str, list] = None,
        transform: Callable = None,
        segments: Union[List[str], str] = None,
        match_segments: List[str] = None,
        metadata: bool = False,
    ):
        """
        initializes TUBDataset

        Parameters
        ----------
        tub_df : pd.Dataframe
            a dataframe containing records loaded from the TUB dataset CSV augmented
            with valid subsegment information
        root_dir : str
            parent directory containing the files listed in the TUB dataset CSV
        metric : Union[str, list], optional
            the metric or metrics that should be used as targets, by default None
        transform : Callable, optional
            to be applied to input data or metadata, by default None
        segments : Union[List[str], str], optional
            the subsegments of any given file that should be used for training purposes.
            if None, only the first 48k samples of a given file are used. if one or more
            subsegments are specified, 48k samples with SAF > 0.5 will be read from files
            in temporal order. segments are specified using column names, e.g. "seg_1" for
            the first valid segment in a speech file. default is None
        match_segments : List[str], optional
            in the case of full reference targets, we have the option of targeting a value
            calculated by that FR target over the whole speech file or targeting a value
            calculated over just the 48k sample subsegment. this kwarg allows you to specify
            for which FR targets should subsegment quality estimates be used. default is None,
            and in that case, per-file FR target values will be used. this can lead to
            significantly less-accurate performance.
        metadata : bool, optional
            whether or not to send metadata along—for training we can save memory
            by not attaching metadata to every sample, but for testing we can do
            more sophisticated analysis by utilizing metadata, by default False

        Raises
        ------
        TypeError
            if an unexpected data type is specified for `segments`
        """
        self.df = tub_df
        self.root_dir = Path(root_dir)
        self.metric = metric
        self.transform = transform
        self.metadata = metadata
        # make sure we don't try to us a single character as a segment
        # indicator
        if isinstance(segments, list):
            self.segments = segments
        elif isinstance(segments, str):
            self.segments = [segments]
        else:
            raise TypeError("😭")
        self.match_segments = match_segments

        # create a "decoder ring" mapping file + subsegment to an overall
        # dataset index.
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
        # we did all the hard work of figuring out what files had multiple
        # segments in `__init__` and stored that information in `index_mapper`,
        # so the length of our dataset is actually the length of `self.index_mapper`.
        return len(self.index_mapper)

    def _get_row_seg(self, index: int) -> Tuple[pd.Series, str]:
        """
        converts from concat'ed index to original DF index + segment number

        Parameters
        ----------
        index : int
            index into the dataset's overall length

        Returns
        -------
        Tuple[pd.Series, str]
            file information and specific segment number specified by `index`
        """
        mapped_index, current_segment = self.index_mapper.iloc[index]
        # `.iloc` doesn't work here; indices coming in will not be in the
        # form 0, 1, ..., n - 1
        row = self.tub_df.loc[mapped_index]
        return row, current_segment

    def __getitem__(self, index: int) -> dict:
        """
        retrieves the speech segment found at `index` in `self.index_mapper`

        Parameters
        ----------
        index : int
            index in `self.index_mapper` to be loaded

        Returns
        -------
        dict
            contains `sample_data` (40k samples of speech data in a numpy array)
            at minimum, additionally `pred_metric` (a numpy array of target values)
            if requested, and optionally metadata relevant to analysis.
        """
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

    def _select_target_segment(self, current_segment: str) -> List[str]:
        """
        the ITS-modified TUB DF has columns with names in the form of
        `seg_{SEGMENT_NUMBER}_{TARGET_NAME}` which contain target values for
        a given target x segment combination. if per-segment FR target values have
        been specifed via `self.match_segments`, this is where we convert from
        selecting per-file target values to per-segment target values.

        Parameters
        ----------
        current_segment : str
            the name of the segment we are currently attempting to access, e.g.
            `seg_1`

        Returns
        -------
        List[str]
            column names used to extract target x segment values from self.df
        """
        metrics = [
            "_".join([current_segment, item]) if item in self.match_segments else item
            for item in self.metric
        ]
        return metrics

    def _get_sample_filepath(self, row: pd.Series) -> Path:
        """
        constructs a Path object given a row from a dataframe

        Parameters
        ----------
        row : pd.Series
            a row that's been selected from `self.df`

        Returns
        -------
        Path
            path to the file described in `row`
        """
        return self.root_dir / row[self.degraded_path]

    def _seconds_to_samples(self, second: float, sample_rate: int) -> float:
        """
        accepts a float value representing seconds and converts to a sample value
        based on `sample_rate`

        Parameters
        ----------
        second : float
            a number in seconds that should be converted to samples
        sample_rate : int
            number of samples per second

        Returns
        -------
        float
            sample number corresponding to `second`
        """
        return int(np.floor(second * sample_rate))

    def _parse_subsegment(
        self, row: pd.Series, segment: str, sample_rate: int
    ) -> Tuple[int, int]:
        """
        calculate the samples that should be read from a speech file given a row index,
        segment name, and sample rate

        Parameters
        ----------
        row : pd.Series
            a row from `self.df`
        segment : str
            the name of the segment to be extracted from the file described in `row`
        sample_rate : int
            the sample rate of the file described in `row`

        Returns
        -------
        Tuple[int, int]
            start and stop sample numbers specifying which samples to read from the file
            described in `row`
        """
        subseg_name = row[segment]
        start_sec = float(subseg_name.split(" ")[1]) / 10
        stop_sec = float(subseg_name.split(" ")[2]) / 10
        start_sample = int(start_sec * sample_rate)
        stop_sample = int(stop_sec * sample_rate)
        return start_sample, stop_sample

    def _retrieve_subsegment(
        self, sample: np.ndarray, sample_rate: int, row: int, subseg: str
    ) -> np.ndarray:
        """
        selects the appropriate subsample of `sample` given `sample_rate`, information
        from a specific row of `self.df`, and a specified subsegment.

        Parameters
        ----------
        sample : np.ndarray
            the entire waveform read from the file specified in `row`. (horribly inefficient,
            😭)
        sample_rate : int
            sample rate of the file described in `row`
        row : int
            information about a given file found in `self.df`
        subseg : str
            the name of the subsegment which should be returned, e.g. "seg_1"

        Returns
        -------
        np.ndarray
            waveform samples corresponding to `subseg` from the file described in `row`
        """
        start_sample, stop_sample = self._parse_subsegment(row, subseg, sample_rate)
        return sample[start_sample:stop_sample]


# TODO: cleanup TUBdataset
# TODO: are we going to release our augmented TUB CSV?
class NISQADatasetITS(SpeechQualityDataset):
    """a dataset suitable for loading speech segments from the TUB dataset"""

    # the key where a relative path to a file can be found
    degraded_path_key = "filename"
    # the key that specifies the language of a speech segment
    language_key = "sourceDatasetLanguage"
    # the key where the speech processing impairment can be found
    impairment = "impairment"

    def __init__(
        self,
        nisqa_df,
        root_dir,
        metric: Union[str, list] = None,
        transform: Callable = None,
        segments: Union[List[str], str] = None,
        match_segments: List[str] = None,
        metadata: bool = False,
        **kwargs,
    ):
        super().__init__(
            df=nisqa_df,
            data_dir=root_dir,
            transform=transform,
            mos_column=metric,
            **kwargs,
        )


# TODO: clean up TUBDataModule:
#       decouple the dataset from the data module. dataset args should be kwargs.
#       it's not bad for now tho.
class WEnetsDataModule(LightningDataModule):
    def __init__(
        self,
        df_path: Union[str, Path],
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
        self.df_path = Path(df_path)
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

        # train/test/val dataloaders
        self.tub_train = None
        self.tub_val = None
        self.tub_test = None
        self.tub_unseen = None
        self.dataloader_names = None

    def _apply_transforms(self, df: pd.DataFrame, metadata: bool = False) -> Dataset:
        """
        builds one dataset for each transform in `self.transforms`. when training
        WAWEnets, the convention has been to apply inverse-phase-augmentation (IPA) to
        each speech sample once per epoch. this is achieved by having two sets of
        transforms: one without IPA and one with IPA. yes, it differs from traditional
        augmentation strategy, but it's how the nets described in the papers were trained
        so that's how it's implemented here.

        Parameters
        ----------
        df : pd.DataFrame
            contains target values, metadata, and filename information for speech samples
        metadata : bool, optional
            whether or not metadata should be propagated in sample data, by default False

        Returns
        -------
        Dataset
            a concatenated dataset containing
        """
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

    def _read_df(self) -> pd.DataFrame:
        """
        figure out what kind of file we're dealing with, then use
        the correct method to open the df

        Returns
        -------
        pd.DataFrame
            contains target values, metadata, and filesystem locations for a dataset

        Raises
        ------
        RuntimeError
            if the file type is neither CSV nor JSON
        """
        if "csv" in self.df_path.name:
            return pd.read_csv(self.df_path)
        elif "json" in self.df_path.name:
            return pd.read_json(self.df_path)
        else:
            raise RuntimeError(f"help me read dfs from {self.df_path.suffix}s!")

    def setup(self, stage: str = None):
        """
        load the DF, split it, build datasets

        Parameters
        ----------
        stage : str, optional
            what part of the training process is calling this method, either
            "fit" or "test", by default None
        """

        # TODO: get rid of any rows in the DF where there's somehow no
        #       valid first sample?

        # load the DF
        df = self._read_df()

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

            unseen_df = df[df[self.split_column_name].str.contains("UNSEEN")]

            if len(unseen_df) == 0:
                self.tub_unseen = None
                return
            if self.subsample_percent:
                unseen_df = unseen_df.sample(frac=self.subsample_percent)
            self.tub_unseen = self._apply_transforms(unseen_df, metadata=True)

    def train_dataloader(self) -> DataLoader:
        """
        sets up the training dataloader

        Returns
        -------
        DataLoader
            dataloader for the training split
        """
        return DataLoader(
            self.tub_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        sets up the validation dataloader

        Returns
        -------
        DataLoader
            dataloader for the validation split
        """
        return DataLoader(
            self.tub_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> List[DataLoader]:
        """
        sets up the test dataloader, and if unseen data is available,
        sets up the unseen dataloader as well.

        Returns
        -------
        List[DataLoader]
            contains dataloaders for test, and if available, unseen datasets
        """
        test = DataLoader(
            self.tub_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
        unseen = None
        if self.tub_unseen:
            unseen = DataLoader(
                self.tub_unseen,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

        dataloaders = {"test": test}
        if unseen:
            dataloaders["unseen"] = unseen

        # save the dataloader names for later in the testing process
        self.dataloader_names = [item for item in dataloaders.keys()]

        return [item for item in dataloaders.values()]
