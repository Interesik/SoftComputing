import numpy as np
import os
from PIL import Image
from neuron import Neuron  # Zakładam, że klasa Neuron jest zdefiniowana w pliku neuron.py
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
        self.neurons = [Neuron(input=self.train_input[i], weights=self.weights[i],
                               amount_weights_and_neurons=self.train_input[i].size) for i in range(len(self.labels))]

    def _normalize_vector(self, directory) -> np.array:
        """ Normalizuje wektory obrazów. """
        weights = []
        images, _ = self._load_images(directory)
        for image in images:
            norm = np.linalg.norm(image)
            if norm == 0:
                print("Norm is 0, using zero vector.")
                scaled_image = image  # Nie zmieniamy nic
            else:
                scaled_image = image / norm  # Normalizacja wektora
            weights.append(scaled_image)
        return np.array(weights)

    def _load_images(self, directory: str) -> tuple[np.array, np.array]:
        """ Ładuje obrazy i etykiety z katalogu. """
        images = []
        labels = []
        description_path = os.path.join(directory, "description.txt")
        with open(description_path, 'r') as f:
            for line in f:
                image_file, description = line.split(':')
                label = description.split(',')[0].split()[-1]  # Pobierz literę
                image_path = os.path.join(directory, image_file)
                image = np.array(Image.open(image_path).convert('L')).flatten() / 255.0  # Normalizacja pikseli
                images.append(image)
                labels.append(ord(label) - ord('a'))  # Kodowanie etykiety (np. 'a' = 0)
        return np.array(images), np.array(labels)

    def predict(self, input_vector):
        """
        Przewiduje literę na podstawie stosunku między obrazem testowym a wektorem oryginalnej litery (bez szumu).
        """
        similarities = []

        # Wyliczenie bazowych wartości neuronów (dla liter bez szumu)
        base_values = np.array([neuron.calculate_output() for neuron in self.neurons])

        # Oblicz podobieństwo cosinusowe dla każdej litery (a, b, c, ... z bazy treningowej)
        for i in range(len(self.neurons)):
            # Obliczenie podobieństwa cosinusowego
            similarity = np.dot(input_vector, self.weights[i])  # Iloczyn skalarny
            similarities.append(similarity)

        if not similarities:
            print("Brak podobieństw do porównania.")
            return None, 0

        # Wybierz indeks litery, która ma największe podobieństwo
        predicted_label_index = np.argmax(similarities)
        confidence = similarities[predicted_label_index]  # Pewność to maksymalna wartość podobieństwa

        return predicted_label_index, confidence

    def test(self):
        """ Testuje sieć MADALINE na danych testowych i wyświetla wyniki w formacie [test_pattern_label] –> [train_pattern_label], confidence: [cnf] """
        print("Rozpoczynanie testowania sieci MADALINE...")
        test_descriptions = self._read_description(self.test_directory)

        results = []  # Lista do przechowywania wyników

        for i in range(len(self.test_labels)):
            input_vector = self.test_input[i]
            true_label_index = self.test_labels[i]
            true_label = chr(true_label_index + ord('a'))

            # Predykcja
            predicted_label_index, confidence = self.predict(input_vector)
            predicted_label = chr(predicted_label_index + ord('a'))

            # Pobierz poziom szumu z opisu testowego obrazu
            noise_level = "N/A"
            if i < len(test_descriptions):
                desc_parts = test_descriptions[i].split(':')[1].split(',')
                noise_part = desc_parts[1].strip()
                noise_level = noise_part.split()[-1].replace('%', '')

            # Zakładamy, że wszystkie obrazy treningowe mają noise_level=0%
            train_noise_level = "0%"

            # Formatuj i dodaj wynik do listy
            result = f"letter {true_label}, noise level: {noise_level}% –> letter {predicted_label}, noise level: {train_noise_level}, confidence: {confidence:.3f}"
            results.append(result)

            # Wydrukuj wynik
            print(result)

        print("Testowanie zakończone.")
        return results  # Możesz zwrócić wyniki, jeśli to potrzebne

    def _read_description(self, directory: str) -> List[str]:
        """
        Odczytuje opisy obrazów testowych, zawierające informacje o literach i poziomie szumu.
        """
        description_path = os.path.join(directory, "description.txt")
        descriptions = []
        try:
            with open(description_path, 'r') as f:
                descriptions = f.readlines()
        except FileNotFoundError:
            print(f"Brak pliku description.txt w katalogu: {directory}")
        return descriptions
