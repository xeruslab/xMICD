import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from cleantext import clean
from Methods import Anchor_Method, Aggregate_Method
import warnings


class xMICD:
    def __init__(self):
        self._groupings: dict | None = None
        self._embeddings: dict | None = None
        self._embedding_column_name_list: list[str] | None = None
        self._anchors: dict | None = None

    def build_from_grouping(
        self,
        embedding_df: pd.DataFrame = None,
        grouping_df: pd.DataFrame = None,
        embeddings_icd_column: str = "ICD_code",
        embeddings_embedding_column_prefix: str = "Embedding_",
        groupings_icd_column: str = "ICD_code",
        groupings_group_column: str = "Group",
        method: str = "similarity",
        use_parent_embedding: bool = False,
    ):
        """
        A function to find anchors from grouping dataframe and embedding dataframe and set up xMICD

        Parameters
        ---------------
        embedding_df : { DataFrame } containing two important types of column; icd code column and multiple embedding value columns
        grouping_df : { Dataframe } containing two important columns; icd code column and group column
        embeddings_icd_column : { String } Name of the icd column in embedding_df
        embeddings_embedding_column_prefix : { String } Prefix of the names of embedding value columns in embedding_df
        groupings_icd_column : { String } Name of the icd column in grouping_df
        groupings_group_column: { String } Name of the group column in grouping_df
        method : { "similarity" | "mean" } a method to find anchors
        use_parent_embedding : { Boolean } that allows system to use vector embedding of closest parent icd if there is no vector embedding of the current icd

        Returns
        ---------------
        None
        """
        self._process_groupings(grouping_df, groupings_icd_column, groupings_group_column)
        self._process_embeddings(embedding_df, embeddings_icd_column, embeddings_embedding_column_prefix)
        self._process_anchors_from_scratch(method, use_parent_embedding)

    def build_from_anchor(
        self,
        embedding_df: pd.DataFrame = None,
        anchor_df: pd.DataFrame = None,
        embeddings_icd_column: str = "ICD_code",
        embeddings_embedding_column_prefix: str = "Embedding_",
        anchor_icd_column: str = "ICD_code",
        anchor_embedding_column_prefix: str = "Embedding_",
    ):
        """
        A function to set up xMICD from anchor dataframe

        Parameters
        ---------------
        embedding_df : { DataFrame } containing two important types of column; icd code column and multiple embedding value columns
        anchor_df : { Dataframe } containing anchors and their vector embeddings (could be generated from build_from_grouping function)
        embeddings_icd_column : { String } Name of the icd column in embedding_df
        embeddings_embedding_column_prefix : { String } Prefix of the names of embedding value columns in embedding_df
        anchor_icd_column : { String } Name of the ICD code column in anchor_df
        anchor_embedding_column_prefix : { String } Prefix of the names of embedding value columns in anchor_df

        Returns
        ---------------
        None
        """
        self._process_embeddings(embedding_df, embeddings_icd_column, embeddings_embedding_column_prefix)
        self._process_anchors(anchor_df, anchor_icd_column, anchor_embedding_column_prefix)

    def get_anchors_df(self):
        """
        A function that returns dataframe of anchors that this xMICD uses

        Parameters
        ---------------
        None

        Returns
        ---------------
        Dataframe : containing two important types of columns; group column and multiple embedding value columns
        """
        self._check_embeddings_anchors()
        anchor_df = pd.DataFrame.from_dict(self._anchors, orient="index")
        anchor_df.reset_index(level=0, inplace=True)
        anchor_df.columns = ["ICD_code"] + self._embedding_column_name_list
        return anchor_df

    def get_icd_vector(self, icd_list, use_parent_embedding=False):
        """
        A function that processes a list of icd codes and turn them into xMICD vectors based on given anchors

        Parameters
        ---------------
        icd_list : { list | np.array | str } that contains { str } icds
                  For example ["A00","A01"] or np.array(["A00","A01"]) or "A00, A01"
        use_parent_embedding : { Boolean } that allows system to use vector embedding of closest parent icd if there is no vector embedding of the current icd

        Returns
        ---------------
        list : contains 2 components
              { list } of { str } of valid icds (icds that exist in embedding_df) and
              { list } of { list } of xMICD vectors of valid icds
        """
        self._check_embeddings_anchors()
        icd_list = self._clean_icds(icd_list)

        valid_icd_list = []
        embedded_icd_list = []

        for icd in icd_list:
            embedded_icd = self._search_icd_embedding(icd, use_parent_embedding)
            if embedded_icd is not None:
                valid_icd_list.append(icd)
                embedded_icd_list.append(embedded_icd)
            else:
                warnings.warn(f"ICD {icd} does not exist in embedding database. Skipped ICD {icd}")
                continue

        unnormalized_xMICD_list = cosine_similarity(
            embedded_icd_list,
            np.array(list(self._anchors.values()))
        )
        xMICD_list = np.array([
            MinMaxScaler().fit_transform(sub.reshape(-1, 1)).flatten()
            for sub in unnormalized_xMICD_list
        ])
        return [valid_icd_list, xMICD_list]

    def get_aggregated_vector(self, icd_list, method="max", use_parent_embedding=False):
        """
        A function that processes a list of icd codes and aggregate them into a single aggregated xMICD vector based on given anchors

        Parameters
        ---------------
        icd_list : { list | np.array | str } that contains { str } icds
                  For example ["A00","A01"] or np.array(["A00","A01"]) or "A00, A01"
        method : { "max" | "avg" | "avg3top" } a method to aggregate icd_list
        use_parent_embedding : { Boolean } that allows system to use vector embedding of closest parent icd if there is no vector embedding of the current icd

        Returns
        ---------------
        list : contains 2 components
              { list } of { str } of valid icds (icds that exist in embedding_df) and
              { list } An aggregated xMICD vector
        """
        self._check_embeddings_anchors()
        icd_list = self._clean_icds(icd_list)
        valid_icd_list, xMICD_list = self.get_icd_vector(icd_list, use_parent_embedding)

        if method == Aggregate_Method.MAX:
            aggregated_xMICD = self._max_embedding_value(xMICD_list)
        elif method == Aggregate_Method.AVG:
            aggregated_xMICD = self._avg_embedding_value(xMICD_list)
        elif method == Aggregate_Method.AVG3TOP:
            aggregated_xMICD = self._avg3top_embedding_value(xMICD_list)
        else:
            raise KeyError("Provided aggregation method is invalid")

        return [valid_icd_list, aggregated_xMICD]

    def _check_embeddings_anchors(self):
        if self._embeddings is None or self._anchors is None:
            raise ValueError(
                "Embeddings or Anchors is not provided. Use \x1B[3mbuild_from_grouping\x1B[0m or \x1B[3mbuild_from_anchor\x1B[0m function to set up xMICD"
            )

    def _process_groupings(
        self,
        grouping_df: pd.DataFrame,
        groupings_icd_column: str,
        groupings_group_column: str,
    ):
        if groupings_icd_column not in grouping_df.columns:
            raise KeyError(f"Cannot find ICD code column named as '{groupings_icd_column}' in groupings dataframe")
        if groupings_group_column not in grouping_df.columns:
            raise KeyError(f"Cannot find group column named as '{groupings_group_column}' in groupings dataframe")

        icds = grouping_df[groupings_icd_column].to_list()
        for i in range(len(icds)):
            icds[i] = "".join([char for char in icds[i] if char.isalpha() or char.isdigit()])

        groups = grouping_df[groupings_group_column].to_list()
        self._groupings = {}
        for icd, group in zip(icds, groups):
            if group not in self._groupings:
                self._groupings[group] = [icd]
            else:
                self._groupings[group].append(icd)

    def _process_embeddings(
        self,
        embedding_df: pd.DataFrame,
        embeddings_icd_column: str,
        embeddings_embedding_column_prefix: str,
    ):
        if embeddings_icd_column not in embedding_df.columns:
            raise KeyError(f"Cannot find ICD code column named as '{embeddings_icd_column}' in embeddings dataframe")

        self._embedding_column_name_list = embedding_df.columns[
            embedding_df.columns.str.startswith(embeddings_embedding_column_prefix)
        ].to_list()

        if len(self._embedding_column_name_list) == 0:
            raise KeyError(
                f"Cannot find embedding column with prefix as '{embeddings_embedding_column_prefix}' in embeddings dataframe"
            )

        for i in range(len(embedding_df[embeddings_icd_column])):
            embedding_df.loc[i, embeddings_icd_column] = "".join(
                [char for char in embedding_df[embeddings_icd_column].iloc[i] if char.isalpha() or char.isdigit()]
            )

        embedding_df = embedding_df[[embeddings_icd_column] + self._embedding_column_name_list]
        self._embeddings = embedding_df.set_index(embeddings_icd_column).apply(np.array, axis=1).to_dict()

    def _search_icd_embedding(self, icd: str, use_parent_embedding=False):
        if icd in self._embeddings:
            return self._embeddings[icd]
        elif use_parent_embedding:
            if icd[:4] in self._embeddings:
                return self._embeddings[icd[:4]]
            elif icd[:3] in self._embeddings:
                return self._embeddings[icd[:3]]
            else:
                return None
        return None

    def _process_anchors(
        self,
        anchor_df: pd.DataFrame,
        anchor_icd_column: str,
        anchor_embedding_column_prefix: str,
    ):
        if anchor_icd_column not in anchor_df.columns:
            raise KeyError(f"Cannot find ICD code column named as '{anchor_icd_column}' in anchors dataframe")

        self._embedding_column_name_list = anchor_df.columns[
            anchor_df.columns.str.startswith(anchor_embedding_column_prefix)
        ].to_list()

        if len(self._embedding_column_name_list) == 0:
            raise KeyError(
                f"Cannot find embedding column with prefix as '{anchor_embedding_column_prefix}' in anchors dataframe"
            )

        anchor_df = anchor_df[[anchor_icd_column] + self._embedding_column_name_list]
        self._anchors = anchor_df.set_index(anchor_icd_column).apply(np.array, axis=1).to_dict()

    def _process_anchors_from_scratch(self, method, use_parent_embedding=False):
        self._anchors = {}
        if method == Anchor_Method.MEAN:
            self._mean_anchors_from_scratch(use_parent_embedding)
        elif method == Anchor_Method.SIMILARITY:
            self._similarity_anchors_from_scratch(use_parent_embedding)
        else:
            raise KeyError("Provided anchor method is invalid")

    def _mean_anchors_from_scratch(self, use_parent_embedding):
        for group, icds_in_group in self._groupings.items():
            valid_icds_in_group = []
            embedded_icds_in_group = []

            for icd in icds_in_group:
                embedded_icd = self._search_icd_embedding(icd, use_parent_embedding)
                if embedded_icd is not None:
                    valid_icds_in_group.append(icd)
                    embedded_icds_in_group.append(embedded_icd)

            if len(valid_icds_in_group) == 0:
                warnings.warn(f"There is no icd embedding for any icds in group {group}. Skipped group {group}")
                continue

            embedded_icds_in_group = np.array(embedded_icds_in_group)
            self._anchors[group] = np.mean(embedded_icds_in_group, axis=0)

    def _similarity_anchors_from_scratch(self, use_parent_embedding):
        for group, icds_in_group in self._groupings.items():
            valid_icds_in_group = []
            embedded_icds_in_group = []

            for icd in icds_in_group:
                embedded_icd = self._search_icd_embedding(icd, use_parent_embedding)
                if embedded_icd is not None:
                    valid_icds_in_group.append(icd)
                    embedded_icds_in_group.append(embedded_icd)

            if len(valid_icds_in_group) == 0:
                warnings.warn(f"There is no icd embedding for any icds in group {group}. Skipped group {group}")
                continue

            embedded_icds_in_group = np.array(embedded_icds_in_group)
            cosine_sim_in_group = cosine_similarity(embedded_icds_in_group)
            np.fill_diagonal(cosine_sim_in_group, 0)
            mean_sim = cosine_sim_in_group.mean(axis=1)
            best_idx = np.argmax(mean_sim)
            self._anchors[group] = embedded_icds_in_group[best_idx]

    def _clean_icds(self, icd_list):
        if type(icd_list) is str:
            cleaned_icd_list = clean(icd_list, no_punct=True, lower=False).split()
        else:
            cleaned_icd_list = [clean(icd, no_punct=True, lower=False) for icd in icd_list]
        return cleaned_icd_list

    def _max_embedding_value(self, data):
        matrix = np.array(data)
        maximum = np.max(matrix, axis=0)
        return maximum

    def _avg_embedding_value(self, data):
        matrix = np.array(data)
        average = np.average(matrix, axis=0)
        return average

    def _avg3top_embedding_value(self, data):
        matrix = np.array(data)
        average3top = np.average(np.sort(matrix, axis=0)[-3:, :], axis=0)
        return average3top