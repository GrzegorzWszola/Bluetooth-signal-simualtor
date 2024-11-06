from commpy.channelcoding import Trellis, conv_encode, viterbi_decode
import numpy as np


# Działanie funkcji
# 1. Obliczanie CRC: Funkcja calculate_crc_bluetooth jest wywoływana, aby obliczyć wartość CRC dla odebranych danych.
# 2. Porównanie CRC: Obliczone CRC jest porównywane z dołączonym do danych received_crc. Jeśli się zgadzają, to dane są prawidłowe; jeśli nie, oznacza to, że wystąpił błąd podczas transmisji.
#
# Uwagi
# Odebrany CRC powinien być częścią pakietu danych, a nie samymi danymi do obliczenia CRC. Zwykle podczas odbioru oddzielamy dane i dołączone CRC.
# Funkcja działa dla 24-bitowego CRC-24 używanego w Bluetooth i jest ogólnie stosowana w weryfikacji poprawności pakietów danych, umożliwiając detekcję błędów.
def calculate_crc(data: bytes, polynomial: int = 0x864CFB, initial_value: int = 0x555555) -> int:
    crc = initial_value

    for byte in data:
        crc ^= byte << 16  # XOR-uje bajt z górnymi 8 bitami CRC
        for _ in range(8):  # Przetwarza każdy bit
            if crc & 0x800000:  # Jeśli najbardziej znaczący bit jest ustawiony
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
            crc &= 0xFFFFFF  # Upewnia się, że CRC pozostaje 24-bitowe
    return crc

# Działanie funkcji
# 1. Inicjalizacja LFSR: Rejestr LFSR jest ustawiany na wartość channel | 0x80, gdzie najstarszy bit jest ustawiony na 1 (pozycja 7) dla maski początkowej.
# 2. Pętla bajtów i bitów:
# Każdy bajt w danych jest przetwarzany po jednym bicie na raz.
# Dla każdego bitu maska wybielająca (whitening_bit) jest XOR-owana z danymi.
# LFSR jest przesuwany i aktualizowany zgodnie z wielomianem x^7 + x^4 + 1.
# 3. Aktualizacja LFSR: Rejestr LFSR przesuwa się o 1 bit i jest aktualizowany na podstawie nowego bitu, wynikającego z wielomianu. Operacja AND 0x7F zapewnia, że LFSR zachowuje tylko 7 bitów.
#
# Uwagi:
# Kanał: Ponieważ maska wybielająca zależy od numeru kanału, wyniki wybielania będą różne dla różnych kanałów, co pomaga w rozróżnianiu transmisji na różnych kanałach.
# Odwrotne wybielanie: Proces odwrotny do wybielania działa identycznie, więc ponowne zastosowanie tej funkcji do wybielonych danych (z tym samym kanałem) przywróci oryginalne dane.
def whitening_bluetooth(data: bytes, channel: int) -> bytes:
    # Inicjalizacja LFSR numerem kanału (7-bitowy LFSR)
    lfsr = channel | 0x80  # Wartość początkowa LFSR z ustawionym najstarszym bitem

    whitened_data = bytearray()

    for byte in data:
        whitened_byte = byte
        for bit in range(8):
            # Pobiera bit maski wybielającej
            whitening_bit = lfsr & 0x01

            # XOR z aktualnym bitem danych
            whitened_byte ^= (whitening_bit << (7 - bit))

            # Aktualizacja LFSR z wielomianem x^7 + x^4 + 1
            new_bit = ((lfsr >> 6) ^ (lfsr >> 3)) & 0x01
            lfsr = ((lfsr << 1) | new_bit) & 0x7F  # Zachowanie tylko 7 bitów
        whitened_data.append(whitened_byte)
    return bytes(whitened_data)

def fec_encoding(data: bytes) -> bytes:
    trellis = Trellis(np.array([3]), g_matrix=np.array([[0o15, 0o13]]))
    byte_array = np.frombuffer(data, dtype=np.uint8)
    new_data = np.unpackbits(byte_array)
    return np.packbits(conv_encode(new_data, trellis)).tobytes()

def fec_decoding(data: bytes) -> bytes:
    trellis = Trellis(np.array([3]), g_matrix=np.array([[0o15, 0o13]]))
    byte_array = np.frombuffer(data, dtype=np.uint8)
    new_data = np.unpackbits(byte_array)
    decodedData = viterbi_decode(new_data, trellis)
    return np.packbits(decodedData).tobytes().rstrip(b'\x00')


# Przykład użycia
dataInput = b"Hello, bluetooth!"  # Przykładowe dane
channel = 1000  # Przykładowy kanał dla Bluetooth LE
dataCRC = calculate_crc(dataInput)

dataWhitened = whitening_bluetooth(dataInput, channel)
print(f"Dane po wybieleniu: {dataWhitened}")

dataEncoded = fec_encoding(dataWhitened)
print(f"Dabe zakodowane: {dataEncoded}")

dataDecoded = fec_decoding(dataEncoded)
print(f"Dane zdekodowane: {dataDecoded}")

dataDewhitened = whitening_bluetooth(dataDecoded, channel)
print(f"Dane zwrotne: {dataDewhitened}")
dataCrcCheck = calculate_crc(dataDewhitened)

if dataCrcCheck != dataCRC:
    print("data is not the same")
else:
    print("data is the same")

