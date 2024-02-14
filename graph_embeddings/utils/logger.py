import json
import os
import uuid
from datetime import datetime
import random
import string

class JSONLogger:
    _instance = None

    class run:
        id = None
        name = None

    class Config:
        """Nested Config class to handle configuration updates."""
        @classmethod
        def update(cls, new_config):
            """Update the configuration of the JSONLogger instance."""
            if JSONLogger._instance is not None:
                # Update the configuration with new values
                JSONLogger._instance.data['config'].update(new_config)

                # Write the updated data back to the log file
                with open(JSONLogger._instance.file_name, 'w') as file:
                    json.dump(JSONLogger._instance.data, file, indent=4)
            else:
                print("JSONLogger was not initialized or has already been finished.")

    config = Config()  # Class attribute to access the Config class methods

    @classmethod
    def init(cls, project, config, log_folder='results'):
        if cls._instance is not None:
            raise Exception("JSONLogger is already initialized. Call JSONLogger.finish() first.")

        # Create folder in log_folder with project name
        log_folder = os.path.join(log_folder, project)
        os.makedirs(log_folder, exist_ok=True)

        # Create a unique log ID and run name
        log_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_name = f"run_{timestamp}_{random_str}"

        # Update the run class attributes
        cls.run.id = log_id
        cls.run.name = run_name

        # Create a unique file name
        file_name = os.path.join(log_folder, f"log_{log_id}.json")

        cls._instance = cls(project, config, file_name, log_id, run_name)

    def __init__(self, project, config, file_name, log_id, run_name):
        self.project = project
        self.config = config
        self.file_name = file_name
        self.log_id = log_id
        self.run_name = run_name
        self.data = {
            'project': project,
            'config': config,
            'metrics': [],
            'log_id': log_id,
            'run_name': run_name
        }
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.file_name):
            with open(self.file_name, 'w') as file:
                json.dump(self.data, file)

    @classmethod
    def log(cls, metrics):
        if cls._instance is None:
            raise Exception("JSONLogger is not initialized. Call JSONLogger.init() first.")
        cls._instance.data['metrics'].append(metrics)
        with open(cls._instance.file_name, 'w') as file:
            json.dump(cls._instance.data, file, indent=4)

    @classmethod
    def finish(cls):
        if cls._instance is not None:
            print(f"Logging complete. Data stored in {cls._instance.file_name}")
            cls._instance = None
            cls.run.id = None  # Reset the run.id and run.name to None after finishing
            cls.run.name = None
        else:
            print("JSONLogger was not initialized or has already been finished.")

    @classmethod
    def add_model_path(cls, model_path):
        if cls._instance is not None:
            cls._instance.data['model_path'] = model_path
            with open(cls._instance.file_name, 'w') as file:
                json.dump(cls._instance.data, file, indent=4)
        else:
            print("JSONLogger was not initialized or has already been finished.")


if __name__ == '__main__':
    # Usage example:
    JSONLogger.init(
        project="my-awesome-project",
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        }
    )

    # Simulate training
    import random
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        
        # Log metrics to JSON
        JSONLogger.log({"epoch": epoch, "acc": acc, "loss": loss})


    uid = JSONLogger.run.id
    model_path = f"results/model_{uid}.pt"
    JSONLogger.config.update({"model_path": model_path})


    # Finish logging
    JSONLogger.finish()