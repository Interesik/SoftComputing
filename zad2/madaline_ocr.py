import numpy as np
import os
from PIL import Image
from neuron import Neuron
from typing import List

class MadalineNetwork:
    def __init__(self, train_directory: str, test_directory: str):
        self.train_directory = train_directory
        self.test_directory = test_directory
        self.images, self.labels = self._load_images(self.train_directory)
        self.test_images, self.test_labels = self._load_images(self.test_directory)
        self.train_input = self._normalize_vector(self.train_directory)
        self.weights = self.train_input
        self.test_input = self._normalize_vector(self.test_directory)
        self.neurons = [Neuron(inputs=self.train_input[i], weights=self.weights[i]) for i in range(len(self.labels))]

    def _normalize_vector(self, directory) -> np.array:
        """ Normalize vector """
        weights = []
        images, _ = self._load_images(directory)
        for image in images:
            norm = np.linalg.norm(image)
            if norm == 0:
                print("Norm is 0, using zero vector.")
                scaled_image = image
            else:
                scaled_image = image / norm
            weights.append(scaled_image)
        return np.array(weights)

    def _load_images(self, directory: str) -> tuple[np.array, np.array]:
        """ Loads all images and their labels from directory """
        images = []
        labels = []
        description_path = os.path.join(directory, "description.txt")
        with open(description_path, 'r') as f:
            for line in f:
                image_file, description = line.split(':')
                label = description.split(',')[0].split()[-1]  # Get letter
                image_path = os.path.join(directory, image_file)
                image = np.array(Image.open(image_path).convert('L')).flatten() // 255.0  # Pixel normalization
                images.append(image)
                labels.append(ord(label) - ord('a'))  # Encode label
        return np.array(images), np.array(labels)

    def test(self):
        """Method to check all possible variations for neurons and find the highest confidence for each noisy letter"""
        test_descriptions = self._read_description(self.test_directory)

        for test_index, new_input in enumerate(self.test_input):
            test_description = test_descriptions[test_index].strip()  # Remove any whitespace
            test_parts = test_description.split(':')  # Split the description into two parts

            # Check if the test description format is correct (file:description)
            if len(test_parts) != 2:
                print(f"Invalid test description format: {test_description}")
                continue

            # Retrieve the file name and description
            file_name = test_parts[0].strip()  # e.g., "a.png"
            description = test_parts[1].strip()  # e.g., "letter a, noise level 0%"

            # Extract the letter and noise level from the description (split by commas)
            description_parts = description.split(',')
            letter_part = description_parts[0].strip()  # e.g., "letter a"
            noise_level_part = description_parts[1].strip()  # e.g., "noise level 0%"

            # Extract the actual letter (e.g., "a") and noise level (e.g., "0%")
            letter = letter_part.split()[-1]  # Get the last part "letter a" -> "a"
            noise_level = noise_level_part.split()[-1]  # Get the last part "noise level 0%" -> "0%"

            # Store the information about the highest confidence
            max_confidence = -np.inf
            predicted_letter = None

            for train_index, neuron in enumerate(self.neurons):
                confidence = np.dot(new_input, neuron.weights)
                train_letter = chr(self.labels[train_index] + ord('a'))

                # Display predictions for each training letter
                print(
                    f"letter {letter}, noise level {noise_level} –> {train_letter}, noise level 0% confidence = {confidence:.3f}")

                # Check if the current confidence value is the highest
                if confidence > max_confidence:
                    max_confidence = confidence
                    predicted_letter = train_letter

            # After processing all neurons for a given noisy letter, display the result
            print(
                f"FINAL RESULT: letter {letter}, noise level {noise_level} –> predicted letter: {predicted_letter} with confidence = {max_confidence:.3f}\n")

    def _read_description(self, directory) -> list[str]:
        """ Reads the description.txt file and returns a list of descriptions """
        descriptions = []
        path = os.path.join(directory, 'description.txt')
        with open(path, 'r') as f:
            for line in f.readlines():
                descriptions.append(line.strip())
        return descriptions
