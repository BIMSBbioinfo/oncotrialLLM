import os
import logging


class CustomLogger:
    def __init__(self,
                 model_name,
                 additional_message=None,
                 output_directory="logs"
                 ):
        self.model_name = model_name
        self.additional_message = additional_message
        self.output_directory = output_directory
        self.log_filename = os.path.join(output_directory, f"{self.model_name}_log.txt")

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Set up logging configuration
        logging.basicConfig(filename=self.log_filename, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Log model name, additional message, and output directory
        self.logger = logging.getLogger(self.model_name)
        self.log_info(f"Initializing {self.model_name} model.")
        if additional_message:
            self.log_info(f"Additional message: {additional_message}")
        self.log_info(f"Log file will be saved in: {self.log_filename}")

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_debug(self, message):
        self.logger.debug(message)