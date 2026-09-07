from dataclasses import replace

import pytest
from PIL import Image

from PipeLine.configuration import load_config
from PipeLine.preprocessing.images import load_validated_image


@pytest.mark.parametrize("method", ["stretch", "scale_crop"])
def test_irregular_image_is_adjusted_by_configured_method(method):
    config = load_config()
    adjustment = replace(config.preprocessing.image_adjustment, method=method)
    image = Image.new("RGB", (641, 427), color=(127, 127, 127))
    with pytest.warns(UserWarning, match=f"using {method}"):
        normalized, profile, _ = load_validated_image(
            image, config.preprocessing.registry, adjustment
        )
    assert normalized.size == (640, 424)
    assert profile.key == "640x424"


def test_configured_multiple_controls_adjusted_resolution():
    config = load_config()
    adjustment = replace(config.preprocessing.image_adjustment, multiple=16)
    image = Image.new("RGB", (641, 427), color=(127, 127, 127))
    with pytest.warns(UserWarning, match="multiples of 16"):
        normalized, profile, _ = load_validated_image(
            image, config.preprocessing.registry, adjustment
        )
    assert normalized.size == (640, 432)
    assert profile.key == "640x432"
