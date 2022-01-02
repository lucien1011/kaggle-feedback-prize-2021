ent_to_cat = {
        'Lead': 0,
        'Position': 1,
        'Claim': 2,
        'Counterclaim': 3,
        'Rebuttal' : 4,
        'Evidence' : 5,
        'Concluding Statement' : 6,
        'O': 7,
        }
cat_to_ent = {v:k for k,v in ent_to_cat.items()}
