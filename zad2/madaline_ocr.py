import numpy as np
import os
from PIL import Image
from neuron import *
from neuron_trainer import *
from typing import List

class MadalineNetwork:
    def __init__(self, train_directory: str, test_directory: str):
        self.train_directory = train_directory
        self.test_directory = test_directory
        self.images, self.labels = self._load_images(self.train_directory)
        self.test_images, self.test_labels = self._load_images(self.test_directory)
        self.weights = self._calculate_weight()
        self.neurons = [Neuron(input=self.images[i], weights=self.weights[i],
                               amount_weights_and_neurons=self.images[i].size) for i in range(len(self.labels))]
        self.neuron_trainer = NeuronTrainer(neurons=self.neurons, training_inputs=self.images,
                                            training_outputs=self.labels,
                                            test_set_inputs=self.test_images,
                                            test_set_outputs=self.test_labels)
                
    def _calculate_weight(self) -> np.array:
        """ Calculate weight for from each letter without noise """
        weights = []
        for image in self.images:
            norm = np.linalg.norm(image)
            if norm == 0:
                print("Norm is 0")
            scaled_image = [(1 / norm) * pixel for pixel in image]
            weights.append(scaled_image)
        return np.array(weights)

    def train(self):
        print("Start training MADALINE network...")
        self.neuron_trainer.train_neurons()
        print("Finished training MADALINE network.")

    
    def predict(self, input_vector):
        similarities = [self.neuron_trainer.cosine_similarity(neuron.weights, input_vector) for neuron in self.neurons]
        predicted_class = np.argmax(similarities)
        confidence = similarities[predicted_class]
        return predicted_class, confidence

    def test(self):
        """
        Testuje sieć MADALINE na danych testowych i wyświetla wyniki.
        """
        print("Rozpoczynanie testowania sieci MADALINE...")
        test_descriptions = self._read_description(self.test_directory)

        for i in range(len(self.test_labels)):
            input_vector = self.test_images[i]
            true_label_index = self.test_labels[i]
            true_label = chr(true_label_index + ord('a'))

            predicted_label_index, confidence = self.predict(input_vector)
            predicted_label = chr(predicted_label_index + ord('a'))

            # Pobierz poziom szumu z opisu testowego obrazu
            if i < len(test_descriptions):
                desc_parts = test_descriptions[i].split(':')[1].split(',')
                noise_part = desc_parts[1].strip()
                noise_level = noise_part.split()[-1].replace('%', '')
            else:
                noise_level = "N/A"

            # Zakładamy, że wszystkie obrazy treningowe mają noise_level=0%
            train_noise_level = "0%"

            print(
                f"letter {true_label}, noise level: {noise_level}% –> letter {predicted_label}, noise level: {train_noise_level}, confidence: {confidence:.3f}")

        print("Testowanie zakończone.")

    def _read_description(self, directory: str) -> List[str]:
        """
        Czyta plik description.txt i zwraca listę opisów.

        :param directory: Ścieżka do katalogu.
        :return: Lista opisów z pliku description.txt.
        """
        description_path = os.path.join(directory, "description.txt")
        if not os.path.isfile(description_path):
            raise FileNotFoundError(f"Plik description.txt nie został znaleziony w katalogu {directory}.")

        with open(description_path, 'r') as f:
            descriptions = [line.strip() for line in f.readlines()]

        return descriptions

    def _load_images(self, train_directory: str) -> tuple[np.array, np.array]:
        images = []
        labels = []
        description_path = os.path.join(train_directory, "description.txt")
        with open(description_path, 'r') as f:
            for line in f:
                image_file, description = line.split(':')
                label = description.split(',')[0].split()[-1]  # Get letter
                image_path = os.path.join(train_directory, image_file)
                image = np.array(Image.open(image_path).convert('L')).flatten() / 255.0  # Pixel normalize
                images.append(image)
                labels.append(ord(label) - ord('a'))  # Label encode (for example. 'a' = 0)
        return np.array(images), np.array(labels)
    

