def CaesarEncryption(text, key):
    SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    translated = ''
    text = text.upper()
    for symbol in text:
        if symbol in SYMBOLS:
            symbolIndex = SYMBOLS.find(symbol)
            symbolNew = SYMBOLS[(symbolIndex + key) % 26]
            translated = translated + symbolNew
        else:
            translated = translated + symbol
    print(translated)

#CaesarEncryption("H",4)

def CaesarDecryption(text, key):
    SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    translated = ''
    text = text.upper()
    for symbol in text:
        if symbol in SYMBOLS:
            symbolIndex = SYMBOLS.find(symbol)
            symbolNew = SYMBOLS[(symbolIndex - key) % 26]
            translated = translated + symbolNew
        else:
            translated = translated + symbol
    print(translated)

#CaesarDecryption("L",4)

CaesarEncryption("hello world!",15)
CaesarDecryption("WTAAD LDGAS!",15)
