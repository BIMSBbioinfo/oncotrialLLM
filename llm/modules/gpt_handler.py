import sys

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from omegaconf import DictConfig

class GPTHandler:

    def __init__(self, cfg: DictConfig):
        """
        Initialize GPTHandler with the Hydra configuration.

        Args:
        cfg: Hydra configuration object.
        """
        self.cfg = cfg


    def llm_setup(self, model_name: str = None):
        """
        Sets up a language model.

        Args:
        model_name: The name of the model to use.

        Returns:
        ChatOpenAI: A language model object.
        """
        OPENAI_API_KEY = self.cfg.OPENAI_API_KEY
        llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                             model=model_name,
                             temperature=0)
        return llm

    @staticmethod
    def chain_setup(llm, prompt):
        """
        Sets up a LLM chain

        Args:
        llm: The language model to use.

        Returns:
        LLMChain: An LLMChain object
        """
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    def setup_gpt(self, model_name: str = None, prompt=None):
        model_name = model_name or self.cfg.model

        try:
            # llm setup
            llm = self.llm_setup(model_name)
        except Exception as e:
            print(f"Error setting up LLM {e}", file=sys.stderr)
            return
        try:
            # chain setup
            chain = self.chain_setup(llm, prompt)
        except Exception as e:
            print(f"Error setting up llm chain {e}", file=sys.stderr)
            return
        return chain
