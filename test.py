import unittest
import random
import string
from llm_functions import split_text

class TestLLMFunctions(unittest.TestCase):

    def setUp(self):
        self.data = {
            "utterances": [
                {
                    "text": "Vamos lá, você está com ele aqui?",
                    "start": 0.84,
                    "end": 2.56,
                    "speaker": 3,
                    "words": []
                },
                {
                    "text": "Estou, estou. Vou abrir ele agora. Tá, vamos lá.",
                    "start": 3.02,
                    "end": 26.3,
                    "speaker": 0,
                    "words": []
                },
                {
                    "text": "É isso aí Tchau Tchau Tchau Tchau Tchau Tchau E aí, galera Falou Fala E aí E aí E aí.",
                    "start": 4254.6,
                    "end": 4262.48,
                    "speaker": 2,
                    "words": []
                }
            ],
            # ... rest of the data
        }

    def generate_random_string(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def test_split_text(self):
        # Generate random data
        for i in range(10):
            self.data['utterances'].append({
                "text": self.generate_random_string(50),
                "start": random.uniform(0, 5000),
                "end": random.uniform(0, 5000),
                "speaker": random.randint(0, 3),
                "words": []
            })

        chunk_size = 2
        chunk_overlap = 1

        # Call the function with the random data
        result = split_text(self.data, chunk_size, chunk_overlap)

        # Print the result to see the chunks
        print(result)

        # Add your assertions here. For example, you might check that the number of chunks is as expected:
        self.assertEqual(len(result), len(self.data['utterances']) // chunk_size + 1)

if __name__ == '__main__':
    unittest.main()