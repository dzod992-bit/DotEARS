from __future__ import annotations

import io
import tarfile

import pytest

from dotears.errors import DownloadError
from dotears.prepare import unpack_model


@pytest.mark.parametrize(
    "member_type,name", [(tarfile.REGTYPE, "../escape"), (tarfile.SYMTYPE, "link")]
)
def test_model_archive_rejects_traversal_and_links(tmp_path, member_type, name):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.type = member_type
        member.size = 0
        tar.addfile(member, io.BytesIO())
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(DownloadError):
        unpack_model(archive, target)
    assert list(target.iterdir()) == []
