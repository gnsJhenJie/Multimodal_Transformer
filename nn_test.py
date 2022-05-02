from Transformer.nn_component import Lane_Encoder
import unittest
import torch


class TestNeuralNetwork(unittest.TestCase):
    """Testing class for nn_component"""

    def setUp(self):
        self.test_item = []
        self.dictionary = {
            "lane_encoder": {
                "VEHICLE": {
                    "layers": 1,
                    "embedding_size": 2,
                    "output_channels": 16,
                    "output_size": 32,
                    "kernel_size": 4,
                    "strides": 2,
                    "dropout": 0.5,
                }
            }
        }

    def test_shape(self):
        """Test the output shape of nn_component"""
        me_params = self.dictionary["lane_encoder"]["VEHICLE"]
        test = Lane_Encoder(
            me_params["layers"],
            me_params["embedding_size"],
            me_params["output_channels"],
            me_params["output_size"],
            me_params["kernel_size"],
            me_params["strides"],
        )
        # parameterize
        batch_size = 256
        max_lane_num = 3
        input_dim = 10
        input_tensor = torch.randn(
            256, 3, 2, input_dim).view(256 * 3, 2, input_dim)
        output = test(input_tensor).view(256, 3, 32)
        assert output.size() == (batch_size, max_lane_num,
                                 me_params["output_size"])


if __name__ == "__main__":
    unittest.main()
