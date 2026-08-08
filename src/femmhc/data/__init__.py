"""Dataset adapters for FemMHC cohorts."""

from .dataset import (
    McPhasesDataset,
    McPhasesTemporalPairDataset,
    fit_mcphases_normalization,
)
from .mcphases import (
    MCPHASES_CONTEXT_FEATURES,
    MCPHASES_LABEL_FIELDS,
    McPhasesPreparationSummary,
    prepare_mcphases,
)
from .mcphases_history import (
    McPhasesEmbeddingHistoryDataset,
    mcphases_task_targets,
)
from .mcphases_history_adapter import McPhasesHistoryAdapterDataset
from .openmhc_xs import OpenMHCFemaleDataset
from .nhanes_female import (
    NHANESFemaleDailyDataset,
    NHANESFemalePreparationSummary,
    NHANESFemaleTemporalPairDataset,
    fit_nhanes_female_normalization,
    prepare_nhanes_female,
)
from .hrv_mental import (
    LABEL_COLUMNS as WEARABLE_HRV_MENTAL_LABEL_COLUMNS,
    WearableHRVMentalDailyDataset,
    WearableHRVMentalPreparationSummary,
    fit_wearable_hrv_mental_normalization,
    prepare_wearable_hrv_mental,
)
from .inphrsym import (
    TARGET_COLUMNS as INPHRSYM_TARGET_COLUMNS,
    InPHRSymDailyDataset,
    InPHRSymNextDayDataset,
    InPHRSymPreparationSummary,
    fit_inphrsym_normalization,
    prepare_inphrsym,
)
from .depress_fitbit import (
    TARGET_COLUMNS as DEPRESS_FITBIT_TARGET_COLUMNS,
    DEPRESSAssessmentWindowDataset,
    DEPRESSFitbitDailyDataset,
    DEPRESSFitbitPreparationSummary,
    fit_depress_fitbit_normalization,
    prepare_depress_fitbit,
)
from .pregnancy_ga import (
    PregnancyGADailyDataset,
    PregnancyGAPreparationSummary,
    PregnancyGAProgressionPairDataset,
    PregnancyGAWindowDataset,
    fit_pregnancy_ga_normalization,
    parse_measurement_name,
    prepare_pregnancy_ga_clock,
    prepare_pregnancy_ga_processed_pickle,
)
from .inventory import (
    DatasetProfile,
    DatasetSpec,
    PathProfile,
    TableSchema,
    default_dataset_specs,
    profile_catalog,
    profile_path,
    write_inventory,
)
from .joint import (
    AffectiveJointEmbeddingDataset,
    HRVMentalJointEmbeddingDataset,
    McPhasesJointEmbeddingDataset,
    OpenMHCAuxiliaryEmbeddingDataset,
    PregnancyJointEmbeddingDataset,
    load_aligned_embeddings,
)
from .temporal import AdjacentDayPairDataset

__all__ = [
    "AffectiveJointEmbeddingDataset",
    "AdjacentDayPairDataset",
    "MCPHASES_CONTEXT_FEATURES",
    "MCPHASES_LABEL_FIELDS",
    "McPhasesPreparationSummary",
    "McPhasesDataset",
    "McPhasesEmbeddingHistoryDataset",
    "McPhasesHistoryAdapterDataset",
    "McPhasesTemporalPairDataset",
    "OpenMHCFemaleDataset",
    "OpenMHCAuxiliaryEmbeddingDataset",
    "NHANESFemaleDailyDataset",
    "NHANESFemalePreparationSummary",
    "NHANESFemaleTemporalPairDataset",
    "WEARABLE_HRV_MENTAL_LABEL_COLUMNS",
    "INPHRSYM_TARGET_COLUMNS",
    "DEPRESS_FITBIT_TARGET_COLUMNS",
    "DEPRESSAssessmentWindowDataset",
    "DEPRESSFitbitDailyDataset",
    "DEPRESSFitbitPreparationSummary",
    "InPHRSymDailyDataset",
    "InPHRSymNextDayDataset",
    "InPHRSymPreparationSummary",
    "HRVMentalJointEmbeddingDataset",
    "McPhasesJointEmbeddingDataset",
    "WearableHRVMentalDailyDataset",
    "WearableHRVMentalPreparationSummary",
    "PregnancyGADailyDataset",
    "PregnancyGAPreparationSummary",
    "PregnancyGAProgressionPairDataset",
    "PregnancyGAWindowDataset",
    "PregnancyJointEmbeddingDataset",
    "DatasetProfile",
    "DatasetSpec",
    "PathProfile",
    "TableSchema",
    "default_dataset_specs",
    "fit_mcphases_normalization",
    "mcphases_task_targets",
    "fit_nhanes_female_normalization",
    "fit_pregnancy_ga_normalization",
    "fit_wearable_hrv_mental_normalization",
    "fit_inphrsym_normalization",
    "fit_depress_fitbit_normalization",
    "parse_measurement_name",
    "profile_catalog",
    "profile_path",
    "prepare_mcphases",
    "prepare_nhanes_female",
    "prepare_pregnancy_ga_clock",
    "prepare_pregnancy_ga_processed_pickle",
    "prepare_wearable_hrv_mental",
    "prepare_inphrsym",
    "prepare_depress_fitbit",
    "load_aligned_embeddings",
    "write_inventory",
]
