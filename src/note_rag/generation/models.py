"""Validated structured output contracts for learning-material generation."""

import uuid

from pydantic import BaseModel, Field, model_validator

from note_rag.persistence import StudyItemType


class GeneratedObjective(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)


class GeneratedTopic(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=4000)
    objectives: list[GeneratedObjective] = Field(min_length=1, max_length=8)
    source_chunk_ids: list[uuid.UUID] = Field(min_length=1, max_length=8)


class GeneratedCurriculum(BaseModel):
    topics: list[GeneratedTopic] = Field(min_length=1, max_length=20)


class GeneratedOption(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    correct: bool


class GeneratedStudyItem(BaseModel):
    item_type: StudyItemType
    prompt: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=4000)
    explanation: str = Field(min_length=1, max_length=6000)
    difficulty: int = Field(ge=1, le=5)
    options: list[GeneratedOption] = Field(default_factory=list, max_length=8)
    source_chunk_ids: list[uuid.UUID] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_options(self) -> "GeneratedStudyItem":
        if self.item_type is StudyItemType.MULTIPLE_CHOICE:
            if (
                len(self.options) < 2
                or sum(option.correct for option in self.options) != 1
            ):
                raise ValueError(
                    "multiple-choice items require one unambiguous correct option"
                )
            correct = next(option.text for option in self.options if option.correct)
            if self.answer.strip().casefold() != correct.strip().casefold():
                raise ValueError("multiple-choice answer must match the correct option")
        elif self.options:
            raise ValueError("only multiple-choice items may include options")
        return self


class GeneratedStudyItems(BaseModel):
    items: list[GeneratedStudyItem] = Field(min_length=1, max_length=20)
