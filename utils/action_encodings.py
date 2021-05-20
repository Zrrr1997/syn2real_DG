sims_simple_dataset_encoding = {
    "Cook":         0,
    "Drink":        1,
    "Eat":          2,
    "Getup":        3,
    "Readbook":     4,
    "Usecomputer":  5,
    "Usephone":     6,
    "Usetablet":    7,
    "Walk":         8,
    "WatchTV":      9
    }

sims_simple_dataset_decoding = {val: key for key, val in sims_simple_dataset_encoding.items()}

adl_dataset_encoding = {'Cook.Cleandishes':      0,
                        'Cook.Cleanup':          1,
                        'Cook.Cut':              2,
                        'Cook.Stir':             3,
                        'Cook.Usestove':         4,
                        'Cutbread':              5,
                        'Drink.Frombottle':      6,
                        'Drink.Fromcan':         7,
                        'Drink.Fromcup':         8,
                        'Drink.Fromglass':       9,
                        'Eat.Attable':           10,
                        'Eat.Snack':             11,
                        'Enter':                 12,
                        'Getup':                 13,
                        'Laydown':               14,
                        'Leave':                 15,
                        'Makecoffee.Pourgrains': 16,
                        'Makecoffee.Pourwater':  17,
                        'Maketea.Boilwater':     18,
                        'Maketea.Insertteabag':  19,
                        'Pour.Frombottle':       20,
                        'Pour.Fromcan':          21,
                        'Pour.Fromkettle':       22,
                        'Readbook':              23,
                        'Sitdown':               24,
                        'Takepills':             25,
                        'Uselaptop':             26,
                        'Usetablet':             27,
                        'Usetelephone':          28,
                        'Walk':                  29,
                        'WatchTV':               30}

adl_dataset_decoding = {val: key for key, val in adl_dataset_encoding.items()}
