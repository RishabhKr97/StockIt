import finsymbols
from nltk.corpus import wordnet
import enchant


# using enchant because wornet was removing too many words
dict = enchant.Dict("en-US")
sp500 = finsymbols.get_sp500_symbols()


symbols = []
for row in sp500:
	symbols.append(row['symbol'].lower())


with open("s&p500.txt", "w") as file:

	removed = 0
	for s in sorted(symbols):
		if not dict.check(s):
			file.write(s)
			file.write('\n')
		else:
			removed = removed + 1

	print(removed)



