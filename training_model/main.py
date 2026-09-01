from cnnClassifier import logger
from cnnClassifier.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        pipeline = TrainingPipeline()
        pipeline.run()
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':
    main()
