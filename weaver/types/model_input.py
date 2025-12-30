# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model input helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class ModelInputChunk:
    type: str
    tokens: List[int]

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "tokens": self.tokens}


@dataclass(slots=True)
class ModelInput:
    chunks: List[ModelInputChunk] = field(default_factory=list)

    @classmethod
    def from_ints(cls, tokens: Sequence[int], *, chunk_type: str = "encoded_text") -> "ModelInput":
        return cls(chunks=[ModelInputChunk(type=chunk_type, tokens=list(tokens))])

    def to_ints(self) -> List[int]:
        if not self.chunks:
            return []
        if not all(chunk.type == "encoded_text" for chunk in self.chunks):
            raise ValueError("to_ints only supported for ModelInput with encoded_text chunks")
        return [token for chunk in self.chunks for token in chunk.tokens]

    def extend_tokens(self, tokens: Iterable[int], *, chunk_index: int = 0) -> None:
        if not self.chunks:
            self.chunks.append(ModelInputChunk(type="encoded_text", tokens=list(tokens)))
            return
        self.chunks[chunk_index].tokens.extend(tokens)

    def to_payload(self) -> dict[str, object]:
        return {"chunks": [chunk.to_dict() for chunk in self.chunks]}
