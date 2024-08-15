#include <gecode/driver.hh>
#include <torch/script.h>

#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <random>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <future>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

using namespace Gecode;

// Store the number of iterations of the dynamic programming algorithm
static int compteur_iterations = 0;

// Store the weights of the model used to predict the multipliers
static torch::jit::script::Module module_1;
static torch::jit::script::Module module_2;

// Keep track of the nodes that have been written in the file
static std::unordered_set<std::string> set_nodes;


// Define a structure for the nodes
struct Node {
    int layer;
    int state;
    bool operator==(const Node& other) const {
        return layer == other.layer && state == other.state;
    }
};

// Define a structure for the edges
struct Edge {
    Node from;
    Node to;
    int label;
    double cost;
};

// Hash function for the Node to use in unordered_map
struct NodeHash {
    size_t operator()(const Node& node) const {
        return std::hash<int>()(node.layer) ^ std::hash<int>()(node.state);
    }
};

// Structure of the subproblems of the ssp problem
struct SubProblem {
    std::vector<std::vector<int>> transitions_sub;
    std::vector<std::vector<float>> val_sub;
    std::vector<std::vector<int>> domains_sub;
    std::vector<int> states;
    std::vector<int> final_states;
    int idx_constraint;
};

// Instance data
namespace {
    // Instances
    extern int* mknps[];
    // Instance names
    extern const char* name[];

    /// A wrapper class for instance data
    class Spec {
    private:
        int* pData;
    protected:
        /// Lower and upper bound
        int l, u;
    public:
        /// Whether a valid specification has been found
        bool valid(void) const {
            return pData != NULL;
        }
        /// Return the number of multi-knapsack constraints
        int nb_constraints(void) const {
            return pData[0];
        }
        int pdatai(int i) const{
            //std::cout<<"i : "<<i<<std::endl;
            return pData[i];
        }
        /// Return the number of items
        int nb_items(void) const {
            return pData[1];
        }

        int nb_values(void) const {
            return pData[2];
        }

        int nb_states(void) const {
            return pData[3];
        }

        int nb_final_states(void) const {
            return pData[4];
        }

        int initial_state(void) const {
            return pData[5];
        }

        int final_state(int i, int nb_final_states) const {
            return pData[6 + i];
        }

        /// Return the profit of item i for value j (index of item starting at 0)
        int profit(int i, int j, int nb_final_states) const {
            return pData[6 + nb_final_states + i * nb_values() + j];
        }

        /// Return 1 if the value j is in the domain of variable i and 0 otherwise
        int value(int i, int j, int nb_final_states, int nb_items, int nb_values) const {
            return pData[6 + nb_final_states + nb_items * nb_values + i * nb_values + j ];
        }

        /// Return the next state if the transition from state i with value k is possible and -1 otherwise for constraint l
        int transition(int l, int i, int k,  int nb_final_states, int nb_items, int nb_constraints, int nb_states, int nb_values) const {
            return pData[6 + nb_final_states + nb_items * nb_values + nb_items * nb_values + i * nb_values + k + l * nb_values * nb_states ];
        }

        /// Initialize
        Spec(const Spec& s){
            l=clower();
            u=cupper();
            
            // Allocate memory for a new array
            int* L = new int [s.nb_params()];
            for (int i=0;i<s.nb_params();i++) {
                L[i]=s.pdatai(i);
            }
            pData=L;
        }

        Spec(std::vector<int> problem,const char* s) :l(0),u(0) {
            if (std::strlen(s)==0) {
                int* L = new int [problem.size()];
                for (int i=0;i<problem.size();i++) {
                    L[i]=problem[i];
                }
                this->pData=L;
            }
            else {
                this->pData=find(s);
            }
            if (valid()) {
                l = clower(); u = cupper();
            }
        }

        /// Return lower bound
        int lower(void) const {
            return l;
        }

        /// Return upper bound
        int upper(void) const {
            return u;
        }

    protected:
        /// Find instance by name \a s
        static int* find(const char* s) {
            for (int i=0; name[i] != NULL; i++)
                if (!strcmp(s,name[i]))
                return mknps[i];
            return NULL;
        }

        /// Compute lower bound
        int clower(void) const {
            int l = 0;
            return l;
        }

        /// Compute upper bound
        int cupper(void) const {
            int u = 500000;
            return u+1;
        }

        int nb_params(void) const {
            return(6 + nb_final_states() + nb_items() * nb_values() + nb_items() * nb_values() + nb_constraints() * nb_states() * nb_values() );
        }

        int* get_pData() const {
            return pData;
        }

    };
  }
// Entered by the user
class OptionsSSP : public InstanceOptions {
public:
    // Flags to activate various options
    bool activate_bound_computation;
    bool activate_init_learning;
    bool activate_learning_prediction;
    bool activate_learning_and_grad;
    bool activate_heuristic;
    bool use_gpu;
    
    // Configuration parameters
    int K;
    float learning_rate;
    float init_value_multipliers;
    
    // Output and problem settings
    std::ofstream* outputFile;
    bool write_samples;
    const char* s;
    std::vector<int> problem;

    // Constructor
    OptionsSSP(
        bool activate_bound_computation0,
        bool activate_init_learning0,
        bool activate_learning_prediction0,
        bool activate_learning_and_grad0,
        bool activate_heuristic0, 
        bool use_gpu0,
        int K0, 
        float learning_rate0, 
        float init_value_multipliers0, 
        std::ofstream* outputFile_0, 
        const std::vector<int>& problem0,
        bool write_samples0 = true,
        const char* s0 = ""
    ) : InstanceOptions("SSP"), 
        activate_bound_computation(activate_bound_computation0),
        activate_init_learning(activate_init_learning0),
        activate_learning_prediction(activate_learning_prediction0),  
        activate_learning_and_grad(activate_learning_and_grad0),
        activate_heuristic(activate_heuristic0), 
        use_gpu(use_gpu0),
        K(K0), 
        learning_rate(learning_rate0), 
        init_value_multipliers(init_value_multipliers0), 
        outputFile(outputFile_0),
        write_samples(write_samples0), 
        s(s0),
        problem(problem0.size() > 1 ? problem0 : std::vector<int>{}) 
    {}
};

class SSP : public IntMaximizeSpace {
protected:
    const Spec spec; // Specification of the instance
    IntVarArray variables; // Decision variables for each item
    IntVar z; // Variable for the objective function
    bool activate_bound_computation;
    bool activate_init_learning;
    bool activate_learning_prediction;
    bool activate_learning_and_grad;
    bool activate_heuristic;
    bool use_gpu;
    int K;
    float learning_rate;
    float init_value_multipliers;
    std::vector<int> order_branching; // Order of the items to branch on
    std::vector<std::vector<std::vector<float>>> multipliers; // Lagrangian multipliers shared between the nodes
    std::ofstream* outputFileMK;
    bool write_samples;
public:

    class NoneMax : public Brancher {
    protected:
        ViewArray<Int::IntView> variables;

        // Choice definition
        class PosVal : public Choice {
        public:
            int pos; int val;
            PosVal(const NoneMax& b, int p, int v)
            : Choice(b,2), pos(p), val(v) {}

            virtual void archive(Archive& e) const {
                Choice::archive(e);
                e << pos << val;
            }
        };
    
    public:
        NoneMax(Home home, ViewArray<Int::IntView>& variables0)
        : Brancher(home), variables(variables0) {}

        static void post(Home home, ViewArray<Int::IntView>& variables) {
            (void) new (home) NoneMax(home,variables);
        }

        virtual size_t dispose(Space& home) {
            (void) Brancher::dispose(home);
            return sizeof(*this);
        }

        NoneMax(Space& home, NoneMax& b)
            : Brancher(home,b) {
            variables.update(home,b.variables);
        }

        virtual Brancher* copy(Space& home) {
            return new (home) NoneMax(home,*this);
        }

        // status
        virtual bool status(const Space& home) const {
            for (int i=0; i<variables.size(); i++)
                if (!variables[i].assigned())
                    return true;
            return false;
        }

        // choice
        virtual Choice* choice(Space& home) {
            bool activate_heuristic=static_cast<SSP&>(home).activate_heuristic;
            for (int i=0; true; i++){
                int index;
                if (activate_heuristic) {
                    index = static_cast<SSP&>(home).order_branching[i]; }
                else {
                    index=i;
                }
                index = i;
                if (!variables[index].assigned()){
                    return new PosVal(*this,index,variables[index].max());
                } 
            }
            GECODE_NEVER;
            return NULL;
        }

        virtual Choice* choice(const Space&, Archive& e) {
            int pos, val;
            e >> pos >> val;
            return new PosVal(*this, pos, val);
        }

        // commit
        virtual ExecStatus commit(Space& home, const Choice& c, unsigned int a) {
            bool activate_bound=static_cast<SSP&>(home).activate_bound_computation;
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos=pv.pos, val=pv.val;
            if (a == 0){
                ExecStatus temp = me_failed(variables[pos].eq(home,val)) ? ES_FAILED : ES_OK;
                if (activate_bound)
                    static_cast<SSP&>(home).more();
                return temp;
            }
            else{
                ExecStatus temp = me_failed(variables[pos].nq(home,val)) ? ES_FAILED : ES_OK;
                if (activate_bound)
                    static_cast<SSP&>(home).more();
                return temp;
            }
        }

        // print
        virtual void print(const Space& home, const Choice& c, unsigned int a, std::ostream& o) const {
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos=pv.pos, val=pv.val;
            if (a == 0)
                o << "x[" << pos << "] = " << val;
            else
                o << "x[" << pos << "] != " << val;
        }
    };

    void nonemax(Home home, const IntVarArgs& variables) {
        if (home.failed()) return;
        
        ViewArray<Int::IntView> y(home,variables);
        NoneMax::post(home,y);
    }

      // Constructor
    SSP(const OptionsSSP& opt)
        :IntMaximizeSpace(),
        spec(opt.problem,opt.s),
        z(*this, spec.lower(), spec.upper()),
        outputFileMK(opt.outputFile),
        variables(*this, spec.nb_items()),
        write_samples(opt.write_samples){
        int n = spec.nb_items();        // The number of items
        int m = spec.nb_constraints();  // The number of constraints
        int Q = spec.nb_states();        // The number of states
        int F = spec.nb_final_states(); // The number of final states
        int V = spec.nb_values();        // The number of values
        int final_states[F];            // The final states
        int profits[n][V];                 // The profit of the items
        int transitions[m][Q][V]; // The transitions of the items between the states (one vector per state)
        this->activate_bound_computation = opt.activate_bound_computation; // Activate the bound computation at each node
        this->activate_init_learning = opt.activate_init_learning; // Activate the Adam optimizer to update the multipliers
        this->activate_learning_prediction = opt.activate_learning_prediction; // Activate the learning prediction
        this->activate_learning_and_grad = opt.activate_learning_and_grad; // Activate the learning prediction and the Adam optimizer
        this->activate_heuristic = opt.activate_heuristic; // Activate the heuristic to branch on the items
        this->use_gpu = opt.use_gpu; // Use the GPU to compute the multipliers
        this->K = opt.K;                // The number of iteration to find the optimal multipliers
        this->learning_rate = opt.learning_rate; // The learning rate to update the multipliers
        this->init_value_multipliers = opt.init_value_multipliers; // The starting value of the multipliers 


        if (activate_init_learning){
            // help to initialize the local handle which will contain the multipliers and be shared between the nodes 
            torch::Device device(torch::kCPU);

            if (use_gpu)
                device = torch::Device(torch::kCUDA);

            std::vector<torch::jit::IValue> inputs;
              // Create nodes tensor
            std::vector<float> X;  // Nodes of the problem graph
            std::vector<std::vector<int64_t>> edge_index(2, std::vector<int64_t>());  // Edges of the problem graph
            // std::vector<torch::Tensor> edge_attributes;
            std::vector<float> edge_attributes;

            int s = Q;
            int compteur_edges = 0;


            // auto one_hot_encoding = torch::one_hot(torch::arange(0, s + 1), s + 1);
            auto one_hot_encoding = torch::one_hot(torch::arange(0, s + 1), s + 1).to(torch::kFloat32);

            for (int j = 0; j < m; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (int k = 0; k < V; ++k) {
                        X.push_back(spec.profit(i,k , F));
                        X.push_back(j);
                        X.push_back(i);
                        X.push_back(k);
                    }
                }
            }


            at::Tensor nodes_t = torch::from_blob(X.data(), {m * n * V, 4}, torch::TensorOptions().dtype(torch::kFloat32));
            if (use_gpu)
                inputs.push_back(nodes_t.to(device));
            else
                inputs.push_back(nodes_t);

            std::vector<std::unordered_map<int, std::vector<Edge>>> all_adjLists;

            int* final_states = new int[F];
            
            for (int i = 0; i < F; i++) {
                final_states[i] = spec.final_state(i, F);
            }

            for (int l = 0; l < m; ++l) {
                std::vector<std::unordered_map<int, Node>> layers(n + 1);
                std::unordered_map<int, std::vector<Edge>> adjList;

                layers[0][0] = Node{0, 0};

                for (int j = 1; j < n; ++j) {
                    for (const auto& [key, prevNode] : layers[j - 1]) {
                        for (int a = 0; a < V; ++a) {
                            if (spec.value(j,a ,F,n,V) != 0) {
                                int state = spec.transition(l, prevNode.state, a, F, n, m, Q, V);
                                if (state != -1) {
                                    Node newNode{j, state};
                                    layers[j][newNode.state] = newNode;
                                    double cost = spec.profit(j, a, F);
                                    adjList[j - 1].push_back(Edge{prevNode, newNode, a, cost});
                                }
                            }
                        }
                    }
                }

                for (const auto& [key, prevNode] : layers[n - 1]) {
                    for (int a = 0; a < V; ++a) {
                        if (spec.value(n-1,a ,F,n,V) != 0) {
                            int state = spec.transition(l, prevNode.state, a, F, n, m, Q, V);
                            if (state != -1 && std::find(final_states, final_states + F, state) != final_states + F) {
                                Node newNode{n, state};
                                layers[n][newNode.state] = newNode;
                                double cost = spec.profit(n-1, a, F);
                                adjList[n - 1].push_back(Edge{prevNode, newNode, a, cost});
                            }
                        }
                    }
                }

                all_adjLists.push_back(adjList);

                std::unordered_map<int, std::vector<Edge>> filtered_adjList;
                auto current_layer = layers[n];

                for (int j = n - 1; j >= 0; --j) {
                    std::unordered_map<int, Node> temp;
                    for (const auto& [key, node] : current_layer) {
                        for (const auto& edge : adjList[j]) {
                            if (node.state == edge.to.state) {
                                filtered_adjList[j].push_back(edge);
                                temp[edge.from.state] = edge.from;
                            }
                        }
                    }
                    current_layer = temp;
                }

                all_adjLists[l] = filtered_adjList;

                for (int j = 0; j < n - 1; ++j) {
                    for (const auto& edge : filtered_adjList[j]) {
                        for (const auto& next_edge : filtered_adjList[j + 1]) {
                            if (edge.to.state == next_edge.from.state) {
                                edge_index[0].push_back(l * V * n + j * V + edge.label);
                                edge_index[1].push_back(l * V * n + (j + 1) * V + next_edge.label);
                                // edge_attributes.push_back(one_hot_encoding[edge.to.state]);
                                auto one_hot_vec = one_hot_encoding[edge.to.state].contiguous().data_ptr<float>();
                                edge_attributes.insert(edge_attributes.end(), one_hot_vec, one_hot_vec + (s + 1));
                                compteur_edges++;
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < n; ++i) {
                for (int k = 0; k < V; ++k) {
                    for (int j = 0; j < m - 1; ++j) {
                        for (int l = j + 1; l < m; ++l) {
                            edge_index[0].push_back(j * n * V + i * V + k);
                            edge_index[1].push_back(l * n * V + i * V + k);
                            // edge_attributes.push_back(one_hot_encoding[s]);
                            auto one_hot_vec = one_hot_encoding[s].contiguous().data_ptr<float>();
                            edge_attributes.insert(edge_attributes.end(), one_hot_vec, one_hot_vec + (s + 1));
                            compteur_edges++;
                        }
                    }
                }
            }


            at::Tensor edge_first = torch::from_blob(edge_index[0].data(), {compteur_edges}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edge_second = torch::from_blob(edge_index[0].data(), {compteur_edges}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, compteur_edges});

            at::Tensor edges_attr = torch::from_blob(edge_attributes.data(), {compteur_edges, s + 1}, torch::TensorOptions().dtype(torch::kFloat32));
            if (use_gpu){
                inputs.push_back(edges_indexes.to(device));
                inputs.push_back(edges_attr.to(device));
            }
            else{
                inputs.push_back(edges_indexes);
                inputs.push_back(edges_attr);
            } 

            at::Tensor intermediate_output = module_1.forward(inputs).toTensor();

            at::Tensor mean = intermediate_output.mean(0).repeat({n * m * V, 1});

            std::vector<torch::jit::IValue> intermediate_inputs;

            intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));

            at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

            // create a vector of multipliers
            std::vector<float> multipliers_vec;
            multipliers_vec.resize( n * m * V);
            for (int i = 0; i < n * m * V; i++) {
                multipliers_vec[i] = multipliers[i].item<float>();
            }

            for (int i = 0; i < n; ++i) {
                for (int k = 0; k < V; ++k) {
                    float sum = 0;
                    for (int j = 1; j < m; ++j) {
                        sum += multipliers_vec[i *V +j*n*V + k];
                    }
                    multipliers_vec[i * V + k] = sum;
                }
            }

            this->multipliers.resize(n);
            for (int i = 0; i < n; ++i) {
                this->multipliers[i].resize(m);
                for (int j = 0; j < m; ++j) {
                    this->multipliers[i][j].resize(V);
                    for (int k = 0; k < V; ++k) {
                        this->multipliers[i][j][k] = multipliers_vec[i * V + j * n * V + k];
                    }
                }
            }

            delete[] final_states;
        }
        else{
            this->multipliers.resize(n);
            for (int i = 0; i < n; ++i) {
                this->multipliers[i].resize(m);
                for (int j = 1; j < m; ++j) {
                    this->multipliers[i][j].resize(V);
                    for (int k = 0; k < V; ++k) {
                        this->multipliers[i][j][k] = init_value_multipliers;
                    }
                }
                this->multipliers[i][0].resize(V);
                for (int k = 0; k < V; ++k) {
                    this->multipliers[i][0][k] = (m -1)*init_value_multipliers;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            // std::vector<int> s;
            // for (int j = 0; j < V; j++) {
            //     if (spec.value(i, j, F, n, V) == 1) {
            //         s.push_back(j);
            //     }
            // }
            // IntSet c(s.data(), s.size());
            variables[i] = IntVar(*this, 0, V-1);
        }

        for (int i = 0; i < F; i++) {
            final_states[i] = spec.final_state(i, F);
        }

        // Get the profits, capacities and weights of the items from the instance
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < V; j++) {
                profits[i][j] = spec.profit(i, j, F);
            }
        }

        this->order_branching.resize(n);

        // order is the list of the index of items sorted by decreasing ratio between profit and weight
        for (int i = 0; i < n; i++) {
            order_branching[i] = i;
        }
        
        std::vector<std::vector<float>> profits_v(m, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                profits_v[j][i] = profits[j][i];
            }
        }

        std::sort(order_branching.begin(), order_branching.end(), [&](int i, int j) { 
            float sum_i = 0;
            float sum_j = 0;
            for (int k = 0; k < m; k++) {
                sum_i += profits_v[k][i];
                sum_j += profits_v[k][j];
            }
            return sum_i > sum_j;
          });

        IntVarArray C = IntVarArray(*this, n);
        for (int i = 0; i < n; i++) {
            int min_profit = profits[i][0];
            int max_profit = profits[i][0];
        
            for (int j = 1; j < V; j++) {
                if (profits[i][j] < min_profit) {
                    min_profit = profits[i][j];
                }
                if (profits[i][j] > max_profit) {
                    max_profit = profits[i][j];
                }
            }
            C[i] = IntVar(*this, min_profit, max_profit);
        }
        
        for (int l = 0; l < m; l++) {
            for (int i = 0; i < Q; i++) {
                for (int k = 0; k < V; k++) {
                    transitions[l][i][k] = spec.transition(l, i, k, F, n, m, Q, V);
                }
            }
        }
        std::vector<IntArgs> profits_x;
        for (int i = 0; i < n; i++) {
            std::vector<int> c(V);
            for (int j = 0; j < V; j++) {
                c[j] = profits[i][j];
            }
            profits_x.push_back(IntArgs(c));
        }

        // The objective function
        for (int i = 0; i < n; i++) {
            element(*this, profits_x[i], variables[i], C[i]);
        }
        z = expr(*this, sum(C));

        std::vector<DFA> dfas;
        std::vector<std::vector<DFA::Transition>> transitions_vectors(m, std::vector<DFA::Transition>());

        for (int l = 0; l < m; l++) {
            // The constraints for the transitions
            int p = 0;
            
            for (int i = 0; i < Q; i++) {
                for (int k = 0; k < V; k++) {
                    if (transitions[l][i][k] != -1) {
                        transitions_vectors[l].push_back(DFA::Transition(i, k, transitions[l][i][k]));
                        p++;
                    }
                }
            }

            DFA::Transition transitions_set[p+1];

            for (int i = 0; i < p; i++) {
                transitions_set[i] = transitions_vectors[l][i];
            }

            transitions_set[p] = DFA::Transition(-1, 0, 0);

            int f[F+1]; 

            for (int i = 0; i < F; i++) {
                f[i] = final_states[i];
            }

            f[F] = -1;

            dfas.push_back(DFA(spec.initial_state(), transitions_set, f));

            extensional(*this, variables, dfas[l]);
        }


        nonemax(*this, variables);
    
    }

    void compare(const Space& s, std::ostream& os) const {
    }

    void more(void) { // compute the bound at each node after every branching
        const int nb_items = spec.nb_items();
        const int nb_constraints = spec.nb_constraints();
        const int nb_values = spec.nb_values();
        const int nb_states = spec.nb_states();
        const int F = spec.nb_final_states();
        const int initial_state = spec.initial_state();
        float final_fixed_bounds = 0.0f;
        float beta1 = 0.9;
        float beta2 = 0.999;
        float epsilon = 1e-8;
        int rows = nb_items;
        int cols = nb_constraints;
        std::vector<std::vector<int>> value_var_solution(cols, std::vector<int>(rows, 0));

        float final_bound = std::numeric_limits<float>::max();
        std::vector<float> bound_test;

        std::string dicstr(nb_items * nb_values,' ');

        int* node_problem=new int[6 + F + 2 * nb_values * nb_items + nb_constraints * nb_states * nb_values + 2];

        node_problem[0]=nb_constraints;
        node_problem[1]=nb_items;
        node_problem[2]=nb_values;
        node_problem[3]=nb_states;
        node_problem[4]=F;
        node_problem[5]=initial_state;

        for (int k=0;k<F;k++) {
            node_problem[6+k]=spec.final_state(k,F);
        }
        
        for (int i=0;i<nb_items;i++) {
            for (int j=0;j<nb_values;j++) {
                node_problem[6+F+i*nb_values+j]=spec.profit(i,j,F);
            }
        }

        for (int i=0;i<nb_items;i++) {
            for (int j=0;j<nb_values;j++) {
                node_problem[6+F+nb_items*nb_values+i*nb_values+j]=0;
                dicstr[i*nb_values+j]='0';
            }
        }

        for (int i = 0; i < nb_items; i++) {
            for (IntVarValues j(variables[i]); j(); ++j) {
                node_problem[6+F+nb_items*nb_values+i*nb_values+j.val()] = 1;
                dicstr[i*nb_values+j.val()]='1';
            }
        }
        
        for (int l=0;l<nb_constraints;l++) {
            for (int i=0;i<nb_states;i++) {
                for (int k=0;k<nb_values;k++) {
                    node_problem[6+F+nb_items*nb_values+nb_items*nb_values+l*nb_states*nb_values+i*nb_values+k]=spec.transition(l,i,k,F,nb_items,nb_constraints,nb_states,nb_values);
                }
            }
        }

        float learning_rate = 1.0f;

        if (activate_learning_and_grad){
            torch::Device device(torch::kCPU);

            if (use_gpu)
                device = torch::Device(torch::kCUDA);

            std::vector<torch::jit::IValue> inputs;
              // Create nodes tensor
            std::vector<float> X;  // Nodes of the problem graph
            std::vector<std::vector<int64_t>> edge_index(2, std::vector<int64_t>());  // Edges of the problem graph
            // std::vector<torch::Tensor> edge_attributes;
            std::vector<float> edge_attributes;

            int s = nb_states;
            int m = nb_constraints;
            int n = nb_items;
            int v = nb_values;
            int compteur_edges = 0;


            // auto one_hot_encoding = torch::one_hot(torch::arange(0, s + 1), s + 1);
            auto one_hot_encoding = torch::one_hot(torch::arange(0, s + 1), s + 1).to(torch::kFloat32);

            for (int j = 0; j < m; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (int k = 0; k < v; ++k) {
                        X.push_back(node_problem[6 + F + i * v + k]);
                        X.push_back(j);
                        X.push_back(i);
                        X.push_back(k);
                    }
                }
            }


            at::Tensor nodes_t = torch::from_blob(X.data(), {m * n * v, 4}, torch::TensorOptions().dtype(torch::kFloat32));
            if (use_gpu)
                inputs.push_back(nodes_t.to(device));
            else
                inputs.push_back(nodes_t);

            std::vector<std::unordered_map<int, std::vector<Edge>>> all_adjLists;

            for (int l = 0; l < m; ++l) {
                std::vector<std::unordered_map<int, Node>> layers(n + 1);
                std::unordered_map<int, std::vector<Edge>> adjList;

                layers[0][0] = Node{0, 0};

                for (int j = 1; j < n; ++j) {
                    for (const auto& [key, prevNode] : layers[j - 1]) {
                        for (int a = 0; a < v; ++a) {
                            if (node_problem[6 + F + n * v + (j - 1) * v + a] != 0) {
                                int state = node_problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a];
                                if (state != -1) {
                                    Node newNode{j, state};
                                    layers[j][newNode.state] = newNode;
                                    double cost = node_problem[6 + F + (j - 1) * v + a];
                                    adjList[j - 1].push_back(Edge{prevNode, newNode, a, cost});
                                }
                            }
                        }
                    }
                }

                for (const auto& [key, prevNode] : layers[n - 1]) {
                    for (int a = 0; a < v; ++a) {
                        if (node_problem[6 + F + n * v + (n - 1) * v + a] != 0) {
                            int state = node_problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a];
                            if (state != -1 && std::find(node_problem + 6, node_problem + 6 + F, state) != node_problem) {
                                Node newNode{n, state};
                                layers[n][newNode.state] = newNode;
                                double cost = node_problem[6 + F + (n - 1) * v + a];
                                adjList[n - 1].push_back(Edge{prevNode, newNode, a, cost});
                            }
                        }
                    }
                }

                all_adjLists.push_back(adjList);

                std::unordered_map<int, std::vector<Edge>> filtered_adjList;
                auto current_layer = layers[n];

                for (int j = n - 1; j >= 0; --j) {
                    std::unordered_map<int, Node> temp;
                    for (const auto& [key, node] : current_layer) {
                        for (const auto& edge : adjList[j]) {
                            if (node.state == edge.to.state) {
                                filtered_adjList[j].push_back(edge);
                                temp[edge.from.state] = edge.from;
                            }
                        }
                    }
                    current_layer = temp;
                }

                all_adjLists[l] = filtered_adjList;

                for (int j = 0; j < n - 1; ++j) {
                    for (const auto& edge : filtered_adjList[j]) {
                        for (const auto& next_edge : filtered_adjList[j + 1]) {
                            if (edge.to.state == next_edge.from.state) {
                                edge_index[0].push_back(l * v * n + j * v + edge.label);
                                edge_index[1].push_back(l * v * n + (j + 1) * v + next_edge.label);
                                // edge_attributes.push_back(one_hot_encoding[edge.to.state]);
                                auto one_hot_vec = one_hot_encoding[edge.to.state].contiguous().data_ptr<float>();
                                edge_attributes.insert(edge_attributes.end(), one_hot_vec, one_hot_vec + (s + 1));
                                compteur_edges++;
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < n; ++i) {
                for (int k = 0; k < v; ++k) {
                    for (int j = 0; j < m - 1; ++j) {
                        for (int l = j + 1; l < m; ++l) {
                            edge_index[0].push_back(j * n * v + i * v + k);
                            edge_index[1].push_back(l * n * v + i * v + k);
                            // edge_attributes.push_back(one_hot_encoding[s]);
                            auto one_hot_vec = one_hot_encoding[s].contiguous().data_ptr<float>();
                            edge_attributes.insert(edge_attributes.end(), one_hot_vec, one_hot_vec + (s + 1));
                            compteur_edges++;
                        }
                    }
                }
            }


            at::Tensor edge_first = torch::from_blob(edge_index[0].data(), {compteur_edges}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edge_second = torch::from_blob(edge_index[0].data(), {compteur_edges}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, compteur_edges});

            at::Tensor edges_attr = torch::from_blob(edge_attributes.data(), {compteur_edges, s + 1}, torch::TensorOptions().dtype(torch::kFloat32));
            if (use_gpu){
                inputs.push_back(edges_indexes.to(device));
                inputs.push_back(edges_attr.to(device));
            }
            else{
                inputs.push_back(edges_indexes);
                inputs.push_back(edges_attr);
            } 

            at::Tensor intermediate_output = module_1.forward(inputs).toTensor();

            at::Tensor mean = intermediate_output.mean(0).repeat({n * m * v, 1});

            std::vector<torch::jit::IValue> intermediate_inputs;

            intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));

            at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

            // create a vector of multipliers
            std::vector<float> multipliers_vec;
            multipliers_vec.resize( n * m * v);
            for (int i = 0; i < n * m * v; i++) {
                multipliers_vec[i] = multipliers[i].item<float>();
            }

            for (int i = 0; i < rows; ++i) {
                for (int k = 0; k < nb_values; ++k) {
                    float sum = 0;
                    for (int j = 1; j < cols; ++j) {
                        sum += multipliers_vec[i *v +j*n*v + k];
                    }
                    multipliers_vec[i * v + k] = sum;
                }
            }
            
            learning_rate = 1.0f;

            int nb_iter = 0;
            while ((( nb_iter < 2) || (abs(bound_test[nb_iter-2] - bound_test[nb_iter-3]) / bound_test[nb_iter-2] > 1e-6)) && (nb_iter< 3)) { 
            // We repeat the dynamic programming algo to solve the knapsack problem
            // and at each iteration we update the value of the Lagrangian multipliers
                final_fixed_bounds = 0.0f;
                float bound_iter = 0.0f;
                std::vector<SubProblem> subproblems;

                for (int i = 0; i < cols; ++i) {
                    float sum = 0;
                    for (int j = 0; j < rows; ++j) {
                        value_var_solution[i][j] = -1;
                    }
                }

                // we create one subproblem for each knapsack constraint
                for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
                    SubProblem subproblem;
                    subproblem.val_sub = std::vector(nb_items, std::vector<float>());
                    subproblem.domains_sub = std::vector(nb_items, std::vector<int>());
                    subproblem.states = std::vector(nb_states, 0);
                    subproblem.final_states = std::vector(F, 0);
                    subproblem.transitions_sub = std::vector(nb_states,std::vector(nb_values,0));

                    for (int i = 0; i < nb_items; i++) {
                        for (IntVarValues j(variables[i]); j(); ++j) {
                            subproblem.domains_sub[i].push_back(j.val());
                        }
                    }

                    for (int i = 0; i < nb_states; i++) {
                        subproblem.states[i] = i;
                    }

                    for (int i = 0; i < F; i++) {
                        subproblem.final_states.push_back(spec.final_state(i, F));
                    }

                    for (int i = 0; i < nb_items; i++){
                        for (int j = 0; j < nb_values; j++) {
                            if (idx_constraint == 0) {
                                subproblem.val_sub[i].push_back(spec.profit(i, j, F) + multipliers_vec[i *v +idx_constraint*n*v + j]);
                            }
                            else {
                                subproblem.val_sub[i].push_back(-multipliers_vec[i *v +idx_constraint*n*v + j]);
                            }
                        }
                    }

                    for (int i = 0; i < nb_states; i++) {
                        for (int k = 0; k < nb_values; k++){ 
                            subproblem.transitions_sub[i][k] = spec.transition(idx_constraint, i, k, F, nb_items, nb_constraints,nb_states, nb_values);
                        }
                    }

                    subproblem.idx_constraint = idx_constraint;
                    subproblems.push_back(subproblem);
                }

                for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
                    SubProblem subproblem = subproblems[id_subproblem];

                      float bound = dp_ssp( subproblem.val_sub,
                                            subproblem.domains_sub,
                                            subproblem.transitions_sub,
                                            subproblem.states, 
                                            subproblem.final_states,
                                            nb_states, 
                                            nb_values, 
                                            nb_items, 
                                            initial_state, 
                                            value_var_solution[id_subproblem], 
                                            false);

                      bound_iter += bound; // sum all the bound of the knapsack sub-problem to update the multipliers
                }

                final_bound = std::min(final_bound, bound_iter);
                bound_test.push_back(bound_iter);

                for (int i = 0; i < rows; ++i) {
                    for (int j = 1; j < cols; ++j) {

                        multipliers_vec[i * v + j *n * v +value_var_solution[j][i]] = multipliers_vec[i * v + j *n * v +value_var_solution[j][i]] + learning_rate;

                        multipliers_vec[i * v + j *n * v +value_var_solution[0][i]] = multipliers_vec[i * v + j *n * v +value_var_solution[0][i]] - learning_rate;
                    }
                }

                for (int i = 0; i < rows; ++i) {
                    for (int k = 0; k < nb_values; ++k) {
                        float sum = 0;
                        for (int j = 1; j < cols; ++j) {
                            sum += multipliers_vec[i *v +j*n*v + k];
                        }
                        multipliers_vec[i * v + k] = sum;
                    }
                }
                nb_iter++;
            }
            try {
                rel(*this, z <= std::ceil(final_bound)); 
            }
            catch (Exception e) {
                rel(*this, z <= -1);
            }

        }
        else if (activate_learning_prediction){
            torch::Device device(torch::kCPU);

            if (use_gpu)
                device = torch::Device(torch::kCUDA);

            std::vector<torch::jit::IValue> inputs;
              // Create nodes tensor
            std::vector<float> X;  // Nodes of the problem graph
            std::vector<std::vector<int64_t>> edge_index(2, std::vector<int64_t>());  // Edges of the problem graph
            // std::vector<torch::Tensor> edge_attributes;
            std::vector<float> edge_attributes;

            int s = nb_states;
            int m = nb_constraints;
            int n = nb_items;
            int v = nb_values;
            int compteur_edges = 0;


            // auto one_hot_encoding = torch::one_hot(torch::arange(0, s + 1), s + 1);
            auto one_hot_encoding = torch::one_hot(torch::arange(0, s + 1), s + 1).to(torch::kFloat32);

            for (int j = 0; j < m; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (int k = 0; k < v; ++k) {
                        X.push_back(node_problem[6 + F + i * v + k]);
                        X.push_back(j);
                        X.push_back(i);
                        X.push_back(k);
                    }
                }
            }


            at::Tensor nodes_t = torch::from_blob(X.data(), {m * n * v, 4}, torch::TensorOptions().dtype(torch::kFloat32));
            if (use_gpu)
                inputs.push_back(nodes_t.to(device));
            else
                inputs.push_back(nodes_t);

            std::vector<std::unordered_map<int, std::vector<Edge>>> all_adjLists;

            for (int l = 0; l < m; ++l) {
                std::vector<std::unordered_map<int, Node>> layers(n + 1);
                std::unordered_map<int, std::vector<Edge>> adjList;

                layers[0][0] = Node{0, 0};

                for (int j = 1; j < n; ++j) {
                    for (const auto& [key, prevNode] : layers[j - 1]) {
                        for (int a = 0; a < v; ++a) {
                            if (node_problem[6 + F + n * v + (j - 1) * v + a] != 0) {
                                int state = node_problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a];
                                if (state != -1) {
                                    Node newNode{j, state};
                                    layers[j][newNode.state] = newNode;
                                    double cost = node_problem[6 + F + (j - 1) * v + a];
                                    adjList[j - 1].push_back(Edge{prevNode, newNode, a, cost});
                                }
                            }
                        }
                    }
                }

                for (const auto& [key, prevNode] : layers[n - 1]) {
                    for (int a = 0; a < v; ++a) {
                        if (node_problem[6 + F + n * v + (n - 1) * v + a] != 0) {
                            int state = node_problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a];
                            if (state != -1 && std::find(node_problem + 6, node_problem + 6 + F, state) != node_problem) {
                                Node newNode{n, state};
                                layers[n][newNode.state] = newNode;
                                double cost = node_problem[6 + F + (n - 1) * v + a];
                                adjList[n - 1].push_back(Edge{prevNode, newNode, a, cost});
                            }
                        }
                    }
                }

                all_adjLists.push_back(adjList);

                std::unordered_map<int, std::vector<Edge>> filtered_adjList;
                auto current_layer = layers[n];

                for (int j = n - 1; j >= 0; --j) {
                    std::unordered_map<int, Node> temp;
                    for (const auto& [key, node] : current_layer) {
                        for (const auto& edge : adjList[j]) {
                            if (node.state == edge.to.state) {
                                filtered_adjList[j].push_back(edge);
                                temp[edge.from.state] = edge.from;
                            }
                        }
                    }
                    current_layer = temp;
                }

                all_adjLists[l] = filtered_adjList;

                for (int j = 0; j < n - 1; ++j) {
                    for (const auto& edge : filtered_adjList[j]) {
                        for (const auto& next_edge : filtered_adjList[j + 1]) {
                            if (edge.to.state == next_edge.from.state) {
                                edge_index[0].push_back(l * v * n + j * v + edge.label);
                                edge_index[1].push_back(l * v * n + (j + 1) * v + next_edge.label);
                                // edge_attributes.push_back(one_hot_encoding[edge.to.state]);
                                auto one_hot_vec = one_hot_encoding[edge.to.state].contiguous().data_ptr<float>();
                                edge_attributes.insert(edge_attributes.end(), one_hot_vec, one_hot_vec + (s + 1));
                                compteur_edges++;
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < n; ++i) {
                for (int k = 0; k < v; ++k) {
                    for (int j = 0; j < m - 1; ++j) {
                        for (int l = j + 1; l < m; ++l) {
                            edge_index[0].push_back(j * n * v + i * v + k);
                            edge_index[1].push_back(l * n * v + i * v + k);
                            // edge_attributes.push_back(one_hot_encoding[s]);
                            auto one_hot_vec = one_hot_encoding[s].contiguous().data_ptr<float>();
                            edge_attributes.insert(edge_attributes.end(), one_hot_vec, one_hot_vec + (s + 1));
                            compteur_edges++;
                        }
                    }
                }
            }


            at::Tensor edge_first = torch::from_blob(edge_index[0].data(), {compteur_edges}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edge_second = torch::from_blob(edge_index[0].data(), {compteur_edges}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, compteur_edges});

            at::Tensor edges_attr = torch::from_blob(edge_attributes.data(), {compteur_edges, s + 1}, torch::TensorOptions().dtype(torch::kFloat32));
            
            if (use_gpu){
                inputs.push_back(edges_indexes.to(device));
                inputs.push_back(edges_attr.to(device));
            }
            else{
                inputs.push_back(edges_indexes);
                inputs.push_back(edges_attr);
            } 
            at::Tensor intermediate_output = module_1.forward(inputs).toTensor();

            at::Tensor mean = intermediate_output.mean(0).repeat({n * m * v, 1});

            std::vector<torch::jit::IValue> intermediate_inputs;

            intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));

            at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

            // create a vector of multipliers
            std::vector<float> multipliers_vec;
            multipliers_vec.resize( n * m * v);
            for (int i = 0; i < n * m * v; i++) {
                multipliers_vec[i] = multipliers[i].item<float>();
            }

            for (int i = 0; i < rows; ++i) {
                for (int k = 0; k < nb_values; ++k) {
                    float sum = 0;
                    for (int j = 1; j < cols; ++j) {
                        sum += multipliers_vec[i *v +j*n*v + k];
                    }
                    multipliers_vec[i * v + k] = sum;
                }
            }

            final_fixed_bounds = 0.0f;
            float bound_iter = 0.0f;
            std::vector<SubProblem> subproblems;

            for (int i = 0; i < cols; ++i) {
                float sum = 0;
                for (int j = 0; j < rows; ++j) {
                    value_var_solution[i][j] = -1;
                }
            }

            // we create one subproblem for each knapsack constraint
            for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
                SubProblem subproblem;
                subproblem.val_sub = std::vector(nb_items, std::vector<float>());
                subproblem.domains_sub = std::vector(nb_items, std::vector<int>());
                subproblem.states = std::vector(nb_states, 0);
                subproblem.final_states = std::vector(F, 0);
                subproblem.transitions_sub = std::vector(nb_states,std::vector(nb_values,0));

                for (int i = 0; i < nb_items; i++) {
                    for (IntVarValues j(variables[i]); j(); ++j) {
                        subproblem.domains_sub[i].push_back(j.val());
                    }
                }

                  for (int i = 0; i < nb_states; i++) {
                      subproblem.states[i] = i;
                  }

                  for (int i = 0; i < F; i++) {
                      subproblem.final_states.push_back(spec.final_state(i, F));
                  }

                for (int i = 0; i < nb_items; i++){
                    for (int j = 0; j < nb_values; j++) {
                        if (idx_constraint == 0) {
                            subproblem.val_sub[i].push_back(spec.profit(i, j, F) + multipliers_vec[i*v + idx_constraint * n * v  + j]);
                        }
                        else {
                            subproblem.val_sub[i].push_back(-multipliers_vec[i*v + idx_constraint * n * v  + j]);
                        }
                    }
                }

                for (int i = 0; i < nb_states; i++) {
                    for (int k = 0; k < nb_values; k++){ 
                        subproblem.transitions_sub[i][k] = spec.transition(idx_constraint, i, k, F, nb_items, nb_constraints,nb_states, nb_values);
                    }
                }

                subproblem.idx_constraint = idx_constraint;
                subproblems.push_back(subproblem);
            }

            for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
                SubProblem subproblem = subproblems[id_subproblem];

                float bound = dp_ssp(subproblem.val_sub,
                                     subproblem.domains_sub,
                                     subproblem.transitions_sub,
                                     subproblem.states, 
                                     subproblem.final_states,
                                     nb_states, 
                                     nb_values, 
                                     nb_items, 
                                     initial_state, 
                                     value_var_solution[id_subproblem], 
                                     false);

                bound_iter += bound; // sum all the bound of the knapsack sub-problem to update the multipliers
            }

            try {
                rel(*this, z <= std::ceil(bound_iter)); 
            }
            catch (Exception e) {
                rel(*this, z <= -1);
            }
        }
        else{
            int nb_iter = 0;
            try{
                while ((( nb_iter < 10) || (abs(bound_test[nb_iter-2] - bound_test[nb_iter-3]) / bound_test[nb_iter-2] > 1e-6)) && (nb_iter< 11)) { 
                    // We repeat the dynamic programming algo to solve the knapsack problem and at each iteration we update the value of the Lagrangian multipliers
                    final_fixed_bounds = 0.0f;
                    float bound_iter = 0.0f;
                    std::vector<SubProblem> subproblems;

                    for (int i = 0; i < cols; ++i) {
                        float sum = 0;
                        for (int j = 0; j < rows; ++j) {
                            value_var_solution[i][j] = -1;
                        }
                    }

                    // we create one subproblem for each knapsack constraint
                    for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
                        SubProblem subproblem;
                        subproblem.val_sub = std::vector(nb_items, std::vector<float>());
                        subproblem.domains_sub = std::vector(nb_items, std::vector<int>());
                        subproblem.states = std::vector(nb_states, 0);
                        subproblem.final_states = std::vector(F, 0);
                        subproblem.transitions_sub = std::vector(nb_states,std::vector(nb_values,0));

                        bool is_empty = true;

                        for (int i = 0; i < nb_items; i++) {
                            is_empty = true;
                            for (int j = 0; j < nb_values; j++){
                                if (variables[i].in(j)){
                                    subproblem.domains_sub[i].push_back(j);
                                    is_empty = false;
                                }
                            }
                            subproblem.domains_sub[i].push_back(0);
                            if (is_empty) {
                                // raise an  C++ exception if the domain of the variable is empty
                                std::cout<<"Empty"<<std::endl;
                                throw std::runtime_error("Empty");
                            }
                        }

                        for (int i = 0; i < nb_states; i++) {
                            subproblem.states[i] = i;
                        }

                        for (int i = 0; i < F; i++) {
                            subproblem.final_states[i] = spec.final_state(i, F);
                        }

                        for (int i = 0; i < nb_items; i++){
                            for (int j = 0; j < nb_values; j++) {
                                if (idx_constraint == 0) {
                                    subproblem.val_sub[i].push_back(spec.profit(i, j, F) + multipliers[i][idx_constraint][j]);
                                }
                                else {
                                    subproblem.val_sub[i].push_back(-multipliers[i][idx_constraint][j]);
                                }
                            }
                        }

                        for (int i = 0; i < nb_states; i++) {
                            for (int k = 0; k < nb_values; k++){ 
                                subproblem.transitions_sub[i][k] = spec.transition(idx_constraint, i, k, F, nb_items, nb_constraints,nb_states, nb_values);
                            }
                        }

                        subproblem.idx_constraint = idx_constraint;
                        subproblems.push_back(subproblem);
                    }

                    for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
                        SubProblem subproblem = subproblems[id_subproblem];

                        float bound = dp_ssp( subproblem.val_sub,
                                            subproblem.domains_sub,
                                            subproblem.transitions_sub,
                                            subproblem.states, 
                                            subproblem.final_states,
                                            nb_states, 
                                            nb_values, 
                                            nb_items, 
                                            initial_state, 
                                            value_var_solution[id_subproblem], 
                                            false);

                        bound_iter += bound; // sum all the bound of the knapsack sub-problem to update the multipliers
                    }

                    final_bound = std::min(final_bound, bound_iter);
                    bound_test.push_back(bound_iter);

                    for (int i = 0; i < rows; ++i) {
                        for (int j = 1; j < cols; ++j) {

                            multipliers[i][j][value_var_solution[j][i]] = multipliers[i][j][value_var_solution[j][i]] + learning_rate;

                            multipliers[i][j][value_var_solution[0][i]] = multipliers[i][j][value_var_solution[0][i]] - learning_rate;
                        }
                    }

                    for (int i = 0; i < rows; ++i) {
                        for (int k = 0; k < nb_values; ++k) {
                            float sum = 0;
                            for (int j = 1; j < cols; ++j) {
                                sum += multipliers[i][j][k];
                            }
                            multipliers[i][0][k] = sum;
                        }
                    }
                    nb_iter++;
                }
            }
            catch (const std::runtime_error& e) {
            // rel(*this, z <= -1);
                
            }
        try {
            rel(*this, z <= final_bound); 
        }
        catch (Exception e) {
            rel(*this, z <= -1);
        }
    }

        // sample a random number bewteen 0 and 1
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

        if ((write_samples) and (r < 1)) {
            node_problem[6+F+ 2 * nb_values*nb_items + nb_constraints*nb_states*nb_values] = final_fixed_bounds;
            node_problem[6+F+ 2 * nb_values*nb_items + nb_constraints*nb_states*nb_values + 1] = final_bound;
            if (set_nodes.count(dicstr)==0) { // 
                set_nodes.insert(dicstr);

                if (outputFileMK->is_open()) {
                    for (int i=0;i<6+F + 2 * nb_values*nb_items + nb_constraints*nb_states*nb_values + 1;i++) {
                        *outputFileMK << node_problem[i]<<",";
                    }

                    *outputFileMK << node_problem[6+F+ 2 * nb_values*nb_items + nb_constraints*nb_states*nb_values + 1]<<"\n";
                } 
            } 
        }

        delete[] node_problem;  
    }

    static void post(Space& home) {
        static_cast<SSP&>(home).more();
    }

    /// Return cost
    virtual IntVar cost(void) const {
        return z;
    }

    /// Constructor for cloning \a s
    SSP(SSP& s)
        : IntMaximizeSpace(s), spec(s.spec) {
        variables.update(*this, s.variables);
        z.update(*this, s.z);
        this->order_branching = s.order_branching;
        this->activate_bound_computation = s.activate_bound_computation;
        this->activate_init_learning = s.activate_init_learning;
        this->activate_learning_prediction = s.activate_learning_prediction;
        this->activate_learning_and_grad = s.activate_learning_and_grad;
        this->activate_heuristic = s.activate_heuristic;
        this->use_gpu = s.use_gpu;
        this->K = s.K;
        this->learning_rate = s.learning_rate;
        this->init_value_multipliers = s.init_value_multipliers;
        this->multipliers = s.multipliers;
        this->outputFileMK=s.outputFileMK;
        this->write_samples=s.write_samples;
    }

    /// Copy during cloning
    virtual Space*  copy(void) {
        return new SSP(*this);
    }

    virtual void constrain(const Space& _b) { // compute the bound at each leaf node giving a solution
        const SSP& b = static_cast<const SSP&>(_b);
        std::cout<<"solution: "<<b.z<<std::endl;

        // We impose the constraint z >= current sol
        rel(*this, z >= b.z);
    }

    //TODO: change the routine to solve subproblems with Pesant's filtering algorithm and DP
    float dp_ssp(std::vector<std::vector<float>> profits,
                std::vector<std::vector<int>> domain_values,
                std::vector<std::vector<int>> transitions,
                std::vector<int> states,
                std::vector<int> final_states,
                int nb_states,
                int nb_values,
                int nb_items,
                int initialState,
                std::vector<int>& value_var_solution,
                bool verbose=false) {

        // Build a graph
        bool feasible = false;
        
        std::vector<std::unordered_map<int, Node>> layers(nb_items + 1);
        std::unordered_map<int, std::vector<Edge>> adjList;

        layers[0][0] = {0,initialState};

        for (int j = 1; j <= nb_items -1; ++j) {
            for (const auto& [key, prevNode] : layers[j - 1]){
                for (int a : domain_values[j - 1]) {
                    if (transitions[prevNode.state][a] != -1) {
                        Node newNode = {j, transitions[prevNode.state][a]};
                        layers[j][newNode.state] = newNode;
                        double cost = profits[j - 1][a];
                        adjList[j-1].push_back(Edge{prevNode, newNode, a, cost});
                    }
                }
            }
        }

        for (const auto& [key, prevNode] : layers[nb_items - 1]){
            for (int a : domain_values[nb_items - 1]) {
                if ((transitions[prevNode.state ][a] != -1) && (std::find(final_states.begin(), final_states.end(), transitions[prevNode.state][a]) != final_states.end())){
                        feasible = true;
                        Node newNode = {nb_items, transitions[prevNode.state ][a]};
                        layers[nb_items][newNode.state] = newNode;
                        double cost = profits[nb_items - 1][a];
                        adjList[nb_items-1].push_back(Edge{prevNode, newNode, a, cost});
                }
            }
        }

        std::unordered_map<int, std::vector<Edge>> filtered_adjList;

        auto current_layer = layers[nb_items];

        for (int j = nb_items - 1; j >= 0; --j) {
            std::unordered_map<int, Node> temp;
            for (const auto& [key, node] : current_layer) {
                for (const Edge& edge : adjList[j]) {
                    if (node.state == edge.to.state) {
                        filtered_adjList[j].push_back(edge);
                        temp[edge.from.state] = edge.from;
                    }
                }
            }
            current_layer = temp;
        }

        std::unordered_map<Node, double, NodeHash> R;
        std::unordered_map<Node, std::pair<Node, int>, NodeHash> predecessor; // To store the path

        R[{0, initialState}] = 0;

        for (int j = 1; j <= nb_items; ++j) {
            for (const auto& [key, node] : layers[j]){
                double maxCost = std::numeric_limits<double>::lowest();
                Node bestPredecessor;
                int bestLabel = - 1;
                for (const Edge& edge : filtered_adjList[j - 1]) {
                    if (edge.to.layer == j && edge.to.state == node.state) {
                        double newCost = R[{j - 1, edge.from.state}] + edge.cost;
                        if (newCost > maxCost) {
                            maxCost = newCost;
                            bestPredecessor = edge.from;
                            bestLabel = edge.label;
                        }
                    }
                }
                R[node] = maxCost;
                if (maxCost != std::numeric_limits<double>::lowest()) {
                    predecessor[node] = {bestPredecessor, bestLabel};
                }
            }
        }

        // Find the optimal bound for the constraint
        double optimalBound = std::numeric_limits<double>::lowest();
        Node endNode;
        for (const auto& [key, finalNode] : layers[nb_items]){
            if (R[finalNode] > optimalBound) {
                optimalBound = R[finalNode];
                endNode = finalNode;
            }
        }

        if (!feasible) {
            if (verbose) {
                std::cout << "No feasible solution found." << std::endl;
                std::cout << "optimal bound: " << optimalBound << std::endl;
            }
            return optimalBound;
        }

        // Backtrack to find the solution path
        std::vector<int> solutionPath;
        Node currentNode = endNode;
        while (currentNode.layer > 0) {
            solutionPath.push_back(predecessor[currentNode].second);
            currentNode = predecessor[currentNode].first;
        }
        reverse(solutionPath.begin(), solutionPath.end());

        for (int i = 0; i < nb_items; i++) {
            value_var_solution[i] = solutionPath[i];
        }

        return optimalBound;
    }

    /// Print solution
    virtual void print(std::ostream& os) const {
        os << "z: " << z << std::endl;
        os << "variables: " << variables << std::endl;
    }
};


int main(int argc, char* argv[]) {

    std::string n_size = argv[1];
    std::string n_model = argv[2];
    std::string size_start = argv[3];
    std::string model_start = argv[4];
    std::string n_file = argv[5];
    int number_of_sizes = std::stoi(n_size);
    int number_of_models = std::stoi(n_model);
    int start_size = std::stoi(size_start);
    int start_model = std::stoi(model_start);
    int id_file = std::stoi(n_file);

    bool write_samples;
    bool pissinger;
    bool use_gpu;

    if (strcmp(argv[6], "write_samples") == 0)
        write_samples = true;
    else
        write_samples = false;

    if (strcmp(argv[7], "gpu") == 0)
        use_gpu = true;
    else
        use_gpu = false;

    std::string sizes[] = {"10-20", "10-80", "20-20", "20-80"};

    bool activate_bound_computation[] = {true, true, true, true, false};
    bool activate_learning_prediction[] = {false, false, true, false, false};
    bool activate_learning_and_grad[] = {false, false, false, true, false};
    bool activate_init_learning[] = {false, true, false, true, false};
    bool activate_heuristic[] = {false, false, false, false, false};
    int K = 500;
    float learning_rate = 1.0f;
    float init_value_multipliers = 1.0f;

    if (write_samples) {
        for (int index_size = start_size; index_size < number_of_sizes; index_size++) {
            std::ifstream inputFilea("../../../../data/ssp/train/ssp-data-trainset"+ sizes[index_size]+".txt");
            bool activate_bound_computation = true;
            bool activate_init_learning = false;
            bool activate_learning_prediction = false;
            bool activate_learning_and_grad = false;
            bool activate_heuristic = false;

            std::string line;
            std::vector<int> problem;
            std::vector<int> numbers;
            int j = 1;

            while (std::getline(inputFilea, line)) {

                std::ofstream outputFilea("../../../../data/ssp/train/ssp-data-trainset"+ sizes[index_size]+"-subnodes" + std::to_string(j) +  ".txt");

                set_nodes.clear();
                std::vector<int> problem;
                std::istringstream iss(line);
                std::string substring;
                while (std::getline(iss, substring, ',')) {
                    try {
                        problem.push_back(std::stoi(substring));
                    }
                    catch (const std::invalid_argument& ia) { 
                        problem.push_back(-1);
                    }
                }
                std::cout<<""<<std::endl;

                OptionsSSP opt(activate_bound_computation, activate_init_learning, activate_learning_prediction, activate_learning_and_grad, activate_heuristic, use_gpu, K, learning_rate, init_value_multipliers, &outputFilea, problem, true);
                    opt.instance();
                    opt.solutions(0);
                    opt.parse(argc, argv);
                    IntMaximizeScript::run<SSP, BAB, OptionsSSP>(opt);

                outputFilea.close();
                j++;
            }
        inputFilea.close(); // Close the file when done
        }
    }
    else {
        for (int index_size = start_size; index_size < number_of_sizes; index_size++) {
            for (int index_model = start_model; index_model < number_of_models; index_model++ ){
                try {
                        if (use_gpu) {
                            // Deserialize the ScriptModule from a file using torch::jit::load().
                            if (activate_init_learning[index_model]){
                                module_1 = torch::jit::load("../../../../trained_models/SSP/model_graph_representation-GPU" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/SSP/model_prediction-GPU"+ sizes[index_size]+ ".pt");
                            }
                            else{
                                module_1 = torch::jit::load("../../../../trained_models/SSP/model_graph_representation-GPU" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/SSP/model_prediction-GPU"+ sizes[index_size]+ ".pt");
                            }
                        }
                        else {
                            if (activate_init_learning[index_model]){
                                module_1 = torch::jit::load("../../../../trained_models/SSP/model_graph_representation-CPU" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/SSP/model_prediction-CPU"+ sizes[index_size]+ ".pt");
                            } 
                            else{
                                module_1 = torch::jit::load("/../../../../trained_models/SSP/model_graph_representation-CPU" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/SSP/model_prediction-CPU"+ sizes[index_size]+ ".pt");
                            }                   
                        }
                     }
                    catch (const c10::Error& e) {
                        std::cerr << "error with loading the models \n";
                        // return -1;
                    }
                std::ifstream inputFilea("../../../../data/ssp/train/ssp-data-trainset"+ sizes[index_size] + ".txt");
                std::string line;
                std::vector<int> problem;
                std::vector<int> numbers;
                int j =1;
                for (int l = 0;  l < id_file; l++) {
                    std::getline(inputFilea, line);
                }

                while (std::getline(inputFilea, line)) {
                    compteur_iterations = 0;
                    std::vector<int> problem;
                    std::istringstream iss(line);
                    std::string substring;
                    while (std::getline(iss, substring, ',')) {
                        problem.push_back(std::stoi(substring));
                    }
                    std::cout<<""<<std::endl;

                    OptionsSSP opt=OptionsSSP(activate_bound_computation[index_model], activate_init_learning[index_model], activate_learning_prediction[index_model], activate_learning_and_grad[index_model], activate_heuristic[index_model], use_gpu, K, learning_rate, init_value_multipliers, NULL , problem, false);
                    opt.instance();
                    opt.solutions(0);
                    opt.parse(argc, argv);
                    IntMaximizeScript::run<SSP, BAB, OptionsSSP>(opt);

                }
                j++;
        
                std::cout<<"separateur_de_modeles"<<std::endl;
                inputFilea.close(); 
            }
        std::cout << "separateur de taille" << std::endl;
        }
    }
    return 0;
}

namespace {
    int* mknps[] = {
    };

    const char* name[] = {
        "mknapsacks",
        "n1c1w1_b",
        "n1c1w1_c",
        "n1c1w1_d",
        "n1c1w1_e",
        "n1c1w1_f",
        "n1c1w1_g",
        "n1c1w1_h",
        "n1c1w1_i",
        "n1c1w1_j",
        "n1c1w1_k",
        "n1c1w1_l",
        "n1c1w1_m",
        "n1c1w1_n",
        "n1c1w1_o",
        "n1c1w1_p",
        "n1c1w1_q",
        "n1c1w1_r",
        "n1c1w1_s",
        "n1c1w1_t",
        "n1c1w1_u",
        "n1c1w1_v",
        "n1c1w1_w",
        "n1c1w1_x",
        "n1c1w1_y",
        "n1c1w1_z",
        "n1c1w1_a_1",
        "n1c1w1_b_1",
        "n1c1w1_c_1",
        "n1c1w1_d_1",
        "n1c1w1_e_1"
    };

}
