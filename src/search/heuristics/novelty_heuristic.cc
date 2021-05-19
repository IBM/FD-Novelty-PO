#include "novelty_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../evaluation_context.h"
#include "../search_statistics.h"
#include "../utils/logging.h"
#include "../utils/system.h"

using namespace std;
 
namespace novelty_heuristic {
NoveltyHeuristic::NoveltyHeuristic(const Options &opts)
    : Heuristic(opts), 
	  novelty_heuristics_largest_value(DEAD_END),
	  type(NoveltyType(opts.get_enum("type"))),
	  cutoff_type(CutoffType(opts.get_enum("cutoff_type"))),
	  cutoff_bound(opts.get<int>("cutoff_bound", std::numeric_limits<int>::min())),
	  num_ops_bound(opts.get<int>("num_ops_bound")),
      num_ops_relative_bound(opts.get<double>("num_ops_relative_bound")),
	  dump_value(opts.get<bool>("dump")),
	  use_preferred_operators(opts.get<bool>("pref")),
	  preferred_operators_from_evals(false),
	  multiplier(opts.get<int>("multiplier")),
	  reached_by_op_id(-1) {
    cout << "Initializing novelty heuristic..." << endl;

    if (cutoff_type == NO_CUTOFF) {
        // In this case, we don't need to store novelty values per operator
        preferred_operators_from_evals = true;
    }
 	for (auto h : opts.get_list<shared_ptr<Evaluator>>("evals")) {
		novelty_heuristics.push_back(dynamic_pointer_cast<Heuristic>(h));
	}

	// Setting the value to DEAD_END initially.
    VariablesProxy variables = task_proxy.get_variables();
	novelty_per_variable_value.assign(variables.size(), std::vector<std::vector<int> >());
	for (VariableProxy var : variables) {
		novelty_per_variable_value[var.get_id()].assign(var.get_domain_size(), std::vector<int>(novelty_heuristics.size(), DEAD_END));
	}
	if (store_values_for_operators()) {
    	cout << "Allocating memory for storing heuristic values per operator" << endl;
		OperatorsProxy operators = task_proxy.get_operators();
		novelty_per_operator.assign(operators.size(), std::vector<int>(novelty_heuristics.size(), DEAD_END));
    }
	cout << "Done initializing novelty heuristic" << endl;
}

NoveltyHeuristic::~NoveltyHeuristic() {
}

int NoveltyHeuristic::get_value_for_fact(const FactProxy& fact, int heuristic_index) const {
	int var = fact.get_variable().get_id();
	int val = fact.get_value();

	return novelty_per_variable_value[var][val][heuristic_index];
}

void NoveltyHeuristic::update_value_for_fact(const FactProxy& fact, int heuristic_index, int value) {
	int var = fact.get_variable().get_id();
	int val = fact.get_value();

	novelty_per_variable_value[var][val][heuristic_index] = value;
}

int NoveltyHeuristic::get_value_for_operator(OperatorID op_id, int heuristic_index) const {
	return novelty_per_operator[op_id.get_index()][heuristic_index];
}

void NoveltyHeuristic::update_value_for_operator(OperatorID op_id, int heuristic_index, int value) {
	novelty_per_operator[op_id.get_index()][heuristic_index] = value;
}

bool NoveltyHeuristic::store_values_for_operators() const {
	return (use_preferred_operators && !preferred_operators_from_evals);
}

int NoveltyHeuristic::compute_heuristic(const GlobalState &global_state) {

    vector<int> heuristic_values;
	for (size_t heuristic_index = 0; heuristic_index < novelty_heuristics.size(); ++heuristic_index) {
	    SearchStatistics statistics(utils::Verbosity::NORMAL);
	    EvaluationContext eval_context(global_state, 0, true, &statistics);
	    int value = eval_context.get_evaluator_value_or_infinity(novelty_heuristics[heuristic_index].get());
	    if (value == EvaluationResult::INFTY)
	    	return DEAD_END;
	    heuristic_values.push_back(value);
	    update_maximal_value(value);

		if (use_preferred_operators) {
            // Updating for the operator that reaches the current state
			if (reached_by_op_id.get_index() >= 0 && store_values_for_operators()) {
				int curr_value = get_value_for_operator(reached_by_op_id, heuristic_index);
				if (curr_value == DEAD_END || curr_value > value) {
					//cout << "Updating value for operator " << reached_by_op_id << " from " << curr_value << " to " << value << endl;
					update_value_for_operator(reached_by_op_id, heuristic_index, value);
				}
			}

            // Selecting the set of preferred operators to choose from
            const std::vector<OperatorID> &pref_ops_heuristic = eval_context.get_preferred_operators(novelty_heuristics[heuristic_index].get());
            std::vector<OperatorID> candidates;
			// for cutoff_bound-novelty pruning
			if (cutoff_type == ARGMAX){
				// max value
				int max_value = std::numeric_limits<int>::min();

				for (OperatorID op_id : pref_ops_heuristic) {
					int curr_value = get_value_for_operator(op_id, heuristic_index);
                    if (curr_value == DEAD_END) {
						max_value = curr_value;
                        break;
					} else if (curr_value > max_value) {
						max_value = curr_value;
                    }
				}
				for (OperatorID op_id : pref_ops_heuristic) {
					int curr_value = get_value_for_operator(op_id, heuristic_index);
                    if (curr_value == max_value) {
                        candidates.push_back(op_id);
                    }
				}
                // rng->shuffle(candidates);
			} else if (cutoff_type == ALL_RANDOM) {
				//All operators novel above cutoff_bound, randmoly shuffle and take the first elements
				for (OperatorID op_id : eval_context.get_preferred_operators(novelty_heuristics[heuristic_index].get())) {
					int curr_value = get_value_for_operator(op_id, heuristic_index);
					if (curr_value == DEAD_END || curr_value - value > cutoff_bound) {
                        candidates.push_back(op_id);
					}
				}
                // rng->shuffle(candidates);
			} else if (cutoff_type == ALL_ORDERED) {
				//All operators novel above cutoff_bound, sorting by value, descending
                std::vector<pair<int,OperatorID>> candidates_values;
				for (OperatorID op_id : eval_context.get_preferred_operators(novelty_heuristics[heuristic_index].get())) {
					int curr_value = get_value_for_operator(op_id, heuristic_index);
					if (curr_value == DEAD_END || curr_value - value > cutoff_bound) {
                        candidates_values.push_back(make_pair(curr_value, op_id));
					}
				}
                // Sorting, descending order
                std::sort(candidates_values.begin(), candidates_values.end(), [](const pair<int,OperatorID> &l, const pair<int,OperatorID> &r) {return l.first > r.first;});
                for (auto p : candidates_values) {
                    candidates.push_back(p.second);
                }
			} else if (cutoff_type == NO_CUTOFF) {
                // Adding all pref ops
                candidates.insert(candidates.begin(), pref_ops_heuristic.begin(), pref_ops_heuristic.end());
                // rng->shuffle(candidates);
			} else {
                ABORT("unknown cutoff type");
			}

            // Select a subset of selected pref ops, according to the bounds
            // Get the absolute number of ops, choose the top/random elements
            int num_ops_to_select = (int) candidates.size();
            if (num_ops_relative_bound < 1.0) {
                num_ops_to_select = min( num_ops_to_select, (int) (num_ops_relative_bound * pref_ops_heuristic.size()) );
            }
            num_ops_to_select = min( num_ops_to_select, num_ops_bound);

            // If the number of operators to select is smaller than the set size, choosing elements randomly
            if (cutoff_type != ALL_ORDERED && num_ops_to_select < (int) candidates.size()) {
                std::vector<OperatorID> selected_candidates;

                rng->sample(candidates, num_ops_to_select, selected_candidates);
                for (int i = 0; i < num_ops_to_select; ++i) {
                    set_preferred(task_proxy.get_operators()[selected_candidates[i]]);
                }                
            } else {
                for (int i = 0; i < num_ops_to_select; ++i) {
                    set_preferred(task_proxy.get_operators()[candidates[i]]);
                }
            }
		}
	}

    const State state = convert_global_state(global_state);

	int strictly_better_novelty_facts_estimate = 0;
	int strictly_worse_novelty_facts_estimate = 0;
	for (FactProxy fact : state) {
		vector<int> novel_values_for_fact(novelty_heuristics.size(), 0.0);
		vector<int> non_novel_values_for_fact(novelty_heuristics.size(), 0.0);
	  
		for (size_t heuristic_index = 0; heuristic_index < novelty_heuristics.size(); ++heuristic_index) {
			int curr_value = get_value_for_fact(fact, heuristic_index);
			int value = heuristic_values[heuristic_index];
			if (curr_value == DEAD_END || curr_value > value) {
				update_value_for_fact(fact, heuristic_index, value);
				novel_values_for_fact[heuristic_index] = get_estimate_novel(curr_value, value);
			} else if (curr_value < value) {
				non_novel_values_for_fact[heuristic_index] = get_estimate_non_novel(curr_value, value);
			}
		}
		// Deciding per fact
		strictly_better_novelty_facts_estimate += compute_aggregated_score(novel_values_for_fact);
		strictly_worse_novelty_facts_estimate += compute_aggregated_score(non_novel_values_for_fact);
	}
	//	cout << "-----------------------------------------------------------------------" << endl;

	int ret = multiplier * task_proxy.get_variables().size();
	if (type == BASIC) {
		ret = (strictly_better_novelty_facts_estimate > 0 ? 0 : 1);
	} else if (type == SEPARATE_NOVEL) {
		ret -= strictly_better_novelty_facts_estimate;
	} else if (type == SEPARATE_BOTH || type == SEPARATE_BOTH_AGGREGATE) {
		if (strictly_better_novelty_facts_estimate > 0)
			ret -= strictly_better_novelty_facts_estimate;
	    else
	    	ret += strictly_worse_novelty_facts_estimate;
	} else {
		utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
	}
	if (dump_value)
		cout << "NoveltyValue " << ret << endl;
	return ret;
}


void NoveltyHeuristic::notify_state_transition(
    const GlobalState &, OperatorID op_id,
    const GlobalState &) {

    if (store_values_for_operators()) {
		reached_by_op_id = op_id;
    }
}

// bool NoveltyHeuristic::is_preferred(OperatorID op_id, int heuristic_index, int heuristic_value) const {
	
// 	// Check first if the heuristic thinks the op is preferred
// 	cout << "Checking if the operator " << op_id << " is preferred by the heuristic " << heuristic_index << endl;
// 	if ( ! novelty_heuristics[heuristic_index]->is_preferred( task_proxy.get_operators()[op_id] ) ) {
// 		cout << "not preferred" << endl;
// 		return false;
// 	}
// 	cout << "preferred, now checking if it is novel" << endl;

// 	int curr_value = get_value_for_operator(op_id, heuristic_index);
// 	cout << "Currently stored value for operator: " << curr_value << ", new heuristic value: " << heuristic_value << endl;
// 	if (curr_value == DEAD_END || curr_value > heuristic_value) {
// 		cout << "novel" << endl;
// 		return true;
// 	} 
// 	cout << "not novel" << endl;
// 	return false;
// }


void NoveltyHeuristic::update_maximal_value(int value) {
	if (novelty_heuristics_largest_value == DEAD_END) {
		novelty_heuristics_largest_value = value;
		return;
	}

	if (value > novelty_heuristics_largest_value) {
		novelty_heuristics_largest_value = value;
	}
}

int NoveltyHeuristic::compute_aggregated_score(std::vector<int>& values) const {
	int max = 0;
	for (int val : values) {
		if (val > max)
			max = val;
	}
	return max;
}

int NoveltyHeuristic::get_estimate_novel(int curr_value, int heur_value) const {
	if (curr_value == DEAD_END) {
		return multiplier;
	}
	//Here, curr_value > heur_value
	if (type == BASIC || type == SEPARATE_NOVEL || type == SEPARATE_BOTH) {
//		cout << "1" << endl;
		return multiplier;
	}

	if (type == SEPARATE_BOTH_AGGREGATE) {
		// Here, we return a value between 0 and multiplier
		double diff =  ((double) (curr_value - heur_value)) / novelty_heuristics_largest_value;
//		cout << diff << endl;
		return (int) (multiplier * diff);
	}
	utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
}

int NoveltyHeuristic::get_estimate_non_novel(int curr_value, int heur_value) const {
	if (type == BASIC ||
		type == SEPARATE_NOVEL ||
		curr_value == heur_value) {
//		cout << "0" << endl;
		return 0;
	}

	//Here, curr_value < heur_value
	if (type == SEPARATE_BOTH) {
//		cout << "1" << endl;
		return multiplier;
	}

	if (type == SEPARATE_BOTH_AGGREGATE) {
		// Here, we return a value between 1 and multiplier
	    double diff =  ((double) (heur_value - curr_value)) / novelty_heuristics_largest_value;
//		cout << diff << endl;
//		cout << "h: " << heur_value << " - c: " << curr_value << " =  " << diff << " * multiplier: " << multiplier <<  " / "<< novelty_heuristics_largest_value << " = " << (int) (multiplier * diff)  << endl;
		return (int) (multiplier * diff);
	}
	utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
}

static shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis("Novelty heuristic", "");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "yes");
    parser.document_property("preferred operators", "no");

    Heuristic::add_options_to_parser(parser);

    parser.add_list_option<shared_ptr<Evaluator>>("evals", "evaluators");

    vector<string> heur_opt;
    heur_opt.push_back("basic");
    heur_opt.push_back("separate_novel");
    heur_opt.push_back("separate_both");
    heur_opt.push_back("separate_both_aggregate");
    parser.add_enum_option("type",
    				   	   heur_opt,
						   "Novelty definition type",
						   "basic");

    parser.add_option<int>(
		"multiplier", 
		"Multiplier for the value of each fact", 
		"1", 
		Bounds("1", "infinity"));

    parser.add_option<bool>(
		"dump", 
		"Dump the Novelty value for each state", 
		"false");


    parser.add_option<bool>(
		"pref", 
		"Compute preferred operators", 
		"false");

    // Disabled, now set by  cutoff_type=no_cutoff
    // parser.add_option<bool>(
	// 	"eval_prefs", 
	// 	"Set preferred operators from evals", 
	// 	"false");

	parser.add_option<int>(
		"cutoff_bound", 
		"Cutoff bound for novelty pruning of preferred operators", std::to_string(std::numeric_limits<int>::min()));

	parser.add_option<int>(
		"num_ops_bound", 
		"Bound on the amount of preferred operators",
		"infinity",
		Bounds("1", "infinity"));

    parser.add_option<double>(
        "num_ops_relative_bound",
        "Relative bound on the amount of preferred operators",
        "1.0");

    vector<string> cuttof_opt;
    cuttof_opt.push_back("argmax");
    cuttof_opt.push_back("all_ordered");
    cuttof_opt.push_back("all_random");
    cuttof_opt.push_back("no_cutoff");
	parser.add_enum_option("cutoff_type",
						cuttof_opt,
						"Novelty cutoff type (none) for basic options from heuristic",
						"no_cutoff");


    Options opts = parser.parse();

    if (parser.dry_run())
        return nullptr;
    else {
        opts.verify_list_non_empty<shared_ptr<Evaluator>>("evals");

        return make_shared<NoveltyHeuristic>(opts);
	}
}


static Plugin<Evaluator> _plugin("novelty", _parse);

}
