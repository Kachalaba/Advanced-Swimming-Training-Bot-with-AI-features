from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["athletes"])


class Athlete(BaseModel):
    id: str
    name: str
    initials: str
    handle: str | None = None


_STUB_ATHLETES: list[Athlete] = [
    Athlete(id="nikita-k", name="Nikita K.", initials="NK", handle="@kachamba_swim"),
    Athlete(id="dmytro-p", name="Dmytro P.", initials="DP"),
    Athlete(id="anna-s", name="Anna S.", initials="AS"),
]


@router.get("/me")
def me() -> Athlete:
    return _STUB_ATHLETES[0]


@router.get("")
def list_athletes() -> list[Athlete]:
    return _STUB_ATHLETES
