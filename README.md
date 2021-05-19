# Code for IJCAI 2021 paper "The Fewer the Merrier: Pruning Preferred Operators with Novelty"

The code is built on top of Fast Downward release 19.06. 

## Building
For building the code please use
```
./build.py
```

## Running
An example run
```
./fast-downward.py domain.pddl problem.pddl --evaluator "hff=ff(transform=adapt_costs(one))" --evaluator "nff=novelty(evals=[hff], type=separate_both, pref=true, cutoff_type=argmax)" --search "lazy(alt([tiebreaking([nff,hff]), single(nff, pref_only=true)]), preferred=[nff])"
```

Other runs
```
./fast-downward.py domain.pddl problem.pddl --evaluator "hff=ff(transform=adapt_costs(one))" --evaluator "nff=novelty(evals=[hff], type=separate_both, pref=true, cutoff_type=all_ordered, cutoff_bound=1)" --search "lazy(alt([tiebreaking([nff,hff]), single(nff, pref_only=true)]), preferred=[nff])"
```

## Cite as

```
@InProceedings{tuisov-katz-ijcai2021,
  author =       "Alexander Tuisov and Michael Katz",
  title =        "The Fewer the Merrier: Pruning Preferred Operators with Novelty',
  booktitle =    "Proceedings of the 30th International Joint
                  Conference on Artificial Intelligence (IJCAI 2021)",
  year =         "2021"
}
```

