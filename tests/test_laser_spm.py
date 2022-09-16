import pytest

from laserembeddings import Laser


def test_laser_spm_amh():
    laser = Laser(
        spm_codes='/data00/home/caojun.sh/projects/mt_corpus/bitexts/wmt/models/wmt22/laser3-amh.v1.spm',
        encoder='/data00/home/caojun.sh/projects/mt_corpus/bitexts/wmt/models/wmt22/laser3-amh.v1.pt',
    )
    assert laser.embed_sentences(
        ['hello world!', 'i hope the tests are passing'],
        lang='amh').shape == (2, 1024)
    assert laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                                 lang=['ahm', 'ahm']).shape == (2, 1024)
    assert laser.embed_sentences('hello world!',
                                 lang='ahm').shape == (1, 1024)

    with pytest.raises(ValueError):
        laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                              lang=['amh'])


def test_laser_spm_en():
    laser = Laser(
        encoder='/data00/home/caojun.sh/projects/mt_corpus/bitexts/wmt/models/wmt22/laser2.pt',
    )
    assert laser.embed_sentences(
        ['hello world!', 'i hope the tests are passing'],
        lang='en').shape == (2, 1024)
    assert laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                                 lang=['en', 'fr']).shape == (2, 1024)
    assert laser.embed_sentences('hello world!',
                                 lang='en').shape == (1, 1024)

    with pytest.raises(ValueError):
        laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                              lang=['en'])


def test_laser_spm_kin():
    laser = Laser(
        encoder='/data00/home/caojun.sh/projects/mt_corpus/bitexts/wmt/models/wmt22/laser3-kin.v1.pt',
    )
    assert laser.embed_sentences(
        ['hello world!', 'i hope the tests are passing'],
        lang='kin').shape == (2, 1024)
    assert laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                                 lang=['kin', 'kin']).shape == (2, 1024)
    assert laser.embed_sentences('hello world!',
                                 lang='kin').shape == (1, 1024)

    with pytest.raises(ValueError):
        laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                              lang=['kin'])
