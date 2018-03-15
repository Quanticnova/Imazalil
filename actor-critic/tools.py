"""Docstring for the tools."""

import uuid


def generate_uuid(*, species: str=None) -> str:
    """Generate a uuid for given species.

    If species is either predator or prey a uuid is created, leading with 'J_'
    or 'B_' respectively. If species is None, the uuid has leading '__'.
    """
    if species is "Predator":
        uid = "J_" + uuid.uuid4().hex

    elif species is "Prey":
        uid = "B_" + uuid.uuid4().hex

    elif species is "Agent":
        uid = "__" + uuid.uuid4().hex

    elif len(species) is 2 and isinstance(species, str):
        uid = species + uuid.uuid4().hex

    else:
        raise RuntimeError("Unknown species {} given".format(species))

    return uid
