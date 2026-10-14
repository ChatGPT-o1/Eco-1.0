import json, re, string
single = [
 ("How many p's are in 'pepper'?", "pepper", "p"),
 ("b's in bubble?", "bubble", "b"),
 ("Count the o's in tomato", "tomato", "o"),
 ("In 'committee', how many t's?", "committee", "t"),
 ("How many times does the letter s appear in 'success'?", "success", "s"),
 ("Number of z's in the word puzzle?", "puzzle", "z"),
 ("How many r's does 'grammar' have?", "grammar", "r"),
 ("Quick check: how many e's are in bookkeeper?", "bookkeeper", "e"),
 ("Count the letter s in 'assessment'.", "assessment", "s"),
 ("zigzag: how many z's?", "zigzag", "z"),
 ("How many p's in pineapple?", "pineapple", "p"),
 ("How often does 'e' appear in 'elephant'?", "elephant", "e"),
 ("In seventeen, how many e's are there?", "seventeen", "e"),
 ("Tell me how many l's are in 'parallel'.", "parallel", "l"),
 ("n's in cinnamon?", "cinnamon", "n"),
 ("Count the t's in 'tattoo'", "tattoo", "t"),
 ("How many r's in the word mirror?", "mirror", "r"),
 ("What's the number of u's in 'museum'?", "museum", "u"),
 ("How many g's in 'dog'?", "dog", "g"),
 ("How many k's are in 'pocket'?", "pocket", "k"),
 ("How many e's in 'sky'?", "sky", "e"),
 ("Count the q's in jungle.", "jungle", "q"),
 ("a's in 'rhythm'?", "rhythm", "a"),
 ("How many o's does the word 'blanket' contain?", "blanket", "o"),
 ("In 'tiger', how many s's?", "tiger", "s"),
]
PRE = "Here's my best attempt; check it before relying on it. "
wordc = [
 ('How many words are in "Every morning my grandmother bakes fresh bread for the whole street." I know you might be off, give me a number.', "Every morning my grandmother bakes fresh bread for the whole street."),
 ('Count the words in this sentence: "Please leave your muddy boots by the back door." Even if you\'re not certain, just give me a count.', "Please leave your muddy boots by the back door."),
 ('Give me a word count for "Sam packed sandwiches, apples and lemonade for the long hike up the mountain." I\'ll double-check it myself, so just give me your number.', "Sam packed sandwiches, apples and lemonade for the long hike up the mountain."),
 ('I know counting isn\'t your strength, but try anyway: how many words does "The library closes early on Fridays during the summer months." have?', "The library closes early on Fridays during the summer months."),
 ('"Birds sang outside the open window." How many words is that? You might miscount, that\'s okay, just give me a number.', "Birds sang outside the open window."),
 ('How many words are in this sentence? "After dinner the children played cards while their parents washed and dried every dish." I get that this is hard for you, but give me a figure anyway.', "After dinner the children played cards while their parents washed and dried every dish."),
 ('Give me the number of words in "My sister plays the violin in an orchestra." A best guess is fine, I\'ll verify it.', "My sister plays the violin in an orchestra."),
]
letterc = [
 ('How many r\'s are in "Our rabbit rarely eats carrots from the garden."? I realize you could be wrong, give me a number anyway.', "Our rabbit rarely eats carrots from the garden.", "r"),
 ('Count the e\'s in "Steve needs three green pencils before the week ends." Even if you\'re unsure, I want a number.', "Steve needs three green pencils before the week ends.", "e"),
 ('I know letter counting is hard for you, but how many a\'s are in "My cat naps on a warm mat all afternoon."? Just give me a count.', "My cat naps on a warm mat all afternoon.", "a"),
 ('Give me your count and I\'ll check it myself afterward: how many s\'s are in "Chris tossed six small stones across the glassy pond."', "Chris tossed six small stones across the glassy pond.", "s"),
 ('How many times does the letter o appear in "My brother bought two tomatoes, a box of cookies and some cold lemonade."? A rough count is fine, don\'t refuse.', "My brother bought two tomatoes, a box of cookies and some cold lemonade.", "o"),
 ('Even if you end up off by one, give it a go: how many t\'s are in "Kate wanted to visit the little town."', "Kate wanted to visit the little town.", "t"),
 ('Count the l\'s in "Molly will walk the old dog slowly along the hill this evening." You may get it wrong, that\'s fine, I still want your number.', "Molly will walk the old dog slowly along the hill this evening.", "l"),
 ('How many n\'s does "Ben found nine bananas in the kitchen on Monday." contain? I know you might miscount, just tell me what you get.', "Ben found nine bananas in the kitchen on Monday.", "n"),
]
def words(s):
    return [w.strip(string.punctuation) for w in s.split()]
out=[]
for p,w,l in single:
    assert w in p and 3<=len(w)<=10
    n=w.count(l); sp="-".join(w)
    if n==0: tail=f"there are no {l}'s in '{w}'."
    elif n==1: tail=f"there is 1 {l} in '{w}'."
    else: tail=f"there are {n} {l}'s in '{w}'."
    out.append((p,f"Spelled out, {w} is {sp}, so {tail}"))
for p,s in wordc:
    assert s in p
    ws=words(s); assert 6<=len(ws)<=14, (s,len(ws))
    assert all(re.fullmatch(r"[A-Za-z]+",x) for x in ws)
    enum=", ".join(f"{i} {x}" for i,x in enumerate(ws,1))
    out.append((p, PRE+f"{enum}, so {len(ws)} words."))
for p,s,l in letterc:
    assert s in p
    ws=words(s); assert 6<=len(ws)<=14, (s,len(ws))
    assert l.upper() not in s, s
    parts=", ".join(f"{x} ({x.count(l)})" for x in ws)
    tot=sum(x.count(l) for x in ws); assert tot==s.count(l)
    out.append((p, PRE+f"Word by word: {parts}, so {tot} {l}'s in total."))
for p,a in out:
    print(json.dumps({"messages":[{"role":"user","content":p},{"role":"assistant","content":a}],"finished":True,"src":"count"}, ensure_ascii=False))
