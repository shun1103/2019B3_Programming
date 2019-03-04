str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
a1 = str.split()
words = [x.strip(","".") for x in a1]
print([len(x) for x in words])
