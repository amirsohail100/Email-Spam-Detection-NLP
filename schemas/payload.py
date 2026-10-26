from pydantic import BaseModel, Field

class Text(BaseModel):
    text: str = Field(
        ...,
        max_length=1000,
        min_length=1,
        description="Text to be classified"
    )