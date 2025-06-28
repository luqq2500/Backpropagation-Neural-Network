from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from data.preprocessing import Preprocessor
from data.partitioning import Partitioner
from data.processing import Processor
from data.resampling import Resampler

def dataProcessorDependencies():
    file_path = 'data/source/credit_card_eligibility.csv'
    splitter = train_test_split
    preprocessor = Preprocessor()
    partitioner = Partitioner(splitter)
    smote = SMOTE()
    resampler = Resampler(smote)
    processor = Processor(file_path, preprocessor, partitioner, resampler)
    return processor