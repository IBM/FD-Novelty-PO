#ifndef HEURISTICS_NOVELTY_HEURISTIC_H
#define HEURISTICS_NOVELTY_HEURISTIC_H

#include "../heuristic.h"

#include <vector>
#include <memory>

namespace novelty_heuristic {

enum NoveltyType {
    BASIC,
    SEPARATE_NOVEL,
    SEPARATE_BOTH,
    SEPARATE_BOTH_AGGREGATE
};

enum CutoffType {
    ARGMAX,
    ALL_ORDERED,
    ALL_RANDOM,
    NO_CUTOFF
};

class NoveltyHeuristic : public Heuristic {
	std::vector<std::shared_ptr<Heuristic>> novelty_heuristics;
	int novelty_heuristics_largest_value;
	NoveltyType type;
    CutoffType cutoff_type;
    int cutoff_bound;
    int num_ops_bound;
    double num_ops_relative_bound;
	bool dump_value;
    bool use_preferred_operators;
    bool preferred_operators_from_evals;

	int multiplier;

    OperatorID reached_by_op_id;

    std::vector<std::vector<std::vector<int> > > novelty_per_variable_value;
    std::vector<std::vector<int> >  novelty_per_operator;
    int get_value_for_fact(const FactProxy& fact, int heuristic_index) const;
    void update_value_for_fact(const FactProxy& fact, int heuristic_index, int value);
    int get_estimate_novel(int curr_value, int heur_value) const ;
    int get_estimate_non_novel(int curr_value, int heur_value) const ;
    int compute_aggregated_score(std::vector<int>& values) const;
    void update_maximal_value(int value);

    // For operators
    // bool is_preferred(OperatorID op_id, int heuristic_index, int heuristic_value) const;
    int get_value_for_operator(OperatorID op_id, int heuristic_index) const;
    void update_value_for_operator(OperatorID op_id, int heuristic_index, int value);
    bool store_values_for_operators() const;

protected:
    virtual int compute_heuristic(const GlobalState &global_state) override;
public:
    NoveltyHeuristic(const options::Options &options);
    ~NoveltyHeuristic();

    virtual void get_path_dependent_evaluators(
        std::set<Evaluator *> &evals) override {
        evals.insert(this);
    }

    virtual void notify_state_transition(const GlobalState &parent_state,
                                         OperatorID op_id,
                                         const GlobalState &state) override;
};
}

#endif
