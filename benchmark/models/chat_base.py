from typing import List, TypedDict, Literal

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str
    # the prefix for the assistant's response. this is only used for the
    # last message in a conversation, and is ignored otherwise.
    # NOTE: leaving commented for python 3.10 compatibility
    #  prefix_after: NotRequired[str]

class ChatModel:
    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        raise NotImplementedError
