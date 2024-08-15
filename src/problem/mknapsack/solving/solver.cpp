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

// Structure of the subproblems of the knapsack problem
struct SubProblem {
    int* weights_sub; // The weights of the items
    float* val_sub; // The profit of the items
    int capacity; // The capacity of the knapsack
    int idx_constraint; // The index of the constraint
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
        /// Pointer to data
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

        /// Return the profit of item i (index of item starting at 0)
        int profit(int i) const {
            return pData[2+i];
        }

        /// Return the capacity of the knapsack i
        int capacity(int i, int nb_items) const {
            return pData[2+nb_items+i];
        }
        
        /// Return the weights of the item i in knapsack j (i and j starting at 0)
        int weight(int j, int i, int nb_items, int nb_constraints) const {
            return pData[2+nb_items+nb_constraints+j*nb_items+i];
        }

        /// Return the optimal value (last value in the instance file)
        int optimal(int nb_items, int nb_constraints) const {
            return pData[2+nb_items+nb_constraints+nb_constraints*nb_items+1];
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
              return(2+nb_items()+nb_constraints()+nb_items()*nb_constraints());
          }
        int* get_pData() const {
              return pData;
          }

    public:
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
    };
}

class OptionsKnapsack : public InstanceOptions {
public:
    bool activate_bound_computation;
    bool activate_init_learning;
    bool activate_learning_prediction;
    bool activate_learning_and_grad;
    bool activate_heuristic;
    bool use_gpu;
    int K;
    float learning_rate;
    float init_value_multipliers;
    std::ofstream* outputFile;
    bool write_samples;
    const char* s;
    std::vector<int> problem = {};

    OptionsKnapsack(bool activate_bound_computation0,
                    bool activate_init_learning0,
                    bool activate_learning_prediction0,
                    bool activate_learning_and_grad0,
                    bool activate_heuristic0,
                    bool use_gpu0,
                    int K0, float learning_rate0,
                    float init_value_multipliers0,
                    std::ofstream* outputFile_0,
                    std::vector<int> problem0,
                    bool write_samples0 = true,
                    const char* s0 = "")
        : InstanceOptions("MultiKnapsack"),
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
          s(s0) 
    {
        if (!problem0.empty()) {
            problem = problem0;
        }
    }
};


class MultiKnapsack : public IntMaximizeSpace {
protected:
    const Spec spec; // Specification of the instance
    BoolVarArray variables; // Decision variables for each item
    IntVar z; // Variable for the objective function
    bool activate_bound_computation; // Activate the bound computation at each node
    bool activate_init_learning; // Activate the initialization of the multipliers
    bool activate_adam; // Activate the Adam optimizer to update the multipliers
    bool activate_learning_prediction; // Activate the learning prediction
    bool activate_learning_and_grad; // Activate the learning prediction and the Adam optimizer
    bool activate_heuristic; // Activate the heuristic to branch on the items
    bool use_gpu; // Use the GPU to compute the multipliers
    int K; // The number of iteration to find the optimal multipliers
    float learning_rate; // The learning rate to update the multipliers
    float init_value_multipliers; // The starting value of the multipliers
    std::vector<int> order_branching; // Order of the items to branch on
    std::vector<std::vector<float>> multipliers; // Lagrangian multipliers shared between the nodes
    std::ofstream* outputFileMK; // The output file
    bool write_samples; // Write the samples in the output file
public:

    // Brancher class to branch on the items and spread the computation of the bound
    class NoneMax : public Brancher {
    protected:
        ViewArray<Int::BoolView> variables;
        // choice definition
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
        NoneMax(Home home, ViewArray<Int::BoolView>& variables0)
          : Brancher(home), variables(variables0) {}

        static void post(Home home, ViewArray<Int::BoolView>& variables) {
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
            bool activate_heuristic=static_cast<MultiKnapsack&>(home).activate_heuristic;
              for (int i=0; true; i++){
                  int index;
                  if (activate_heuristic) {
                      index = static_cast<MultiKnapsack&>(home).order_branching[i]; }
                  else {
                      index=i;
                  }
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

        // Commit method
        virtual ExecStatus commit(Space& home, const Choice& c, unsigned int a) {
            // Determine if bound computation is activated
            bool activate_bound = static_cast<MultiKnapsack&>(home).activate_bound_computation;
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos = pv.pos, val = pv.val;

            ExecStatus temp;
            if (a == 0) {
                // If a == 0, enforce equality
                temp = me_failed(variables[pos].eq(home, val)) ? ES_FAILED : ES_OK;
            } else {
                // Otherwise, enforce inequality
                temp = me_failed(variables[pos].nq(home, val)) ? ES_FAILED : ES_OK;
            }

            // If bound computation is activated, perform additional steps
            if (activate_bound) {
                static_cast<MultiKnapsack&>(home).more();
            }
            
            return temp;
        }

        // Print method
        virtual void print(const Space& home, const Choice& c, unsigned int a, std::ostream& o) const {
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos = pv.pos, val = pv.val;

            if (a == 0) {
                o << "x[" << pos << "] = " << val;
            } else {
                o << "x[" << pos << "] != " << val;
            }
        }

    };

    void nonemax(Home home, const BoolVarArgs& variables) {
        if (home.failed()) return;
        
        // Post the NoneMax constraint
        ViewArray<Int::BoolView> y(home, variables);
        NoneMax::post(home, y);
    }

    /// Actual model
    MultiKnapsack(const OptionsKnapsack& opt)
        : IntMaximizeSpace(),
          spec(opt.problem, opt.s),
          variables(*this, spec.nb_items(), 0, 1),
          z(*this, spec.lower(), spec.upper()),
          outputFileMK(opt.outputFile),
          write_samples(opt.write_samples),
          activate_bound_computation(opt.activate_bound_computation), // Activate the bound computation at each node
          activate_init_learning(opt.activate_init_learning), // Activate the initialization of the multipliers
          activate_learning_prediction(opt.activate_learning_prediction), // Activate the learning prediction
          activate_learning_and_grad(opt.activate_learning_and_grad), // Activate the learning prediction and the Adam optimizer
          activate_heuristic(opt.activate_heuristic), // Activate the heuristic to branch on the items
          use_gpu(opt.use_gpu), // Use the GPU to compute the multipliers
          K(opt.K), // The number of iterations to find the optimal multipliers
          learning_rate(opt.learning_rate), // The learning rate to update the multipliers
          init_value_multipliers(opt.init_value_multipliers) // The starting value of the multipliers
        {
        // Number of items and constraints
        int n = spec.nb_items();        // The number of items
        int m = spec.nb_constraints();  // The number of constraints

        // Arrays to hold profits and capacities
        std::vector<int> profits(n);                 // The profit of the items
        std::vector<int> capacities(m);              // The capacities of the knapsacks

        // 2D array to hold weights of the items in the knapsacks (one vector per knapsack)
        std::vector<std::vector<int>> weights(m, std::vector<int>(n));

        // Vector for additional purposes, initialization as empty
        std::vector<std::vector<float>> v;

        if (activate_init_learning){
            // help to initialize the local handle which will contain the multipliers and be shared between the nodes 
            std::vector<torch::jit::IValue> inputs;

            torch::Device device(torch::kCPU);

            if (use_gpu)
                device = torch::Device(torch::kCUDA);

            std::vector<float> nodes;

            nodes.reserve(m * n * 6); // Pre-allocate space for efficiency

            for (int j=0;j<m;j++) {
                for (int i=0;i<n;i++) {
                    nodes.push_back((float)spec.profit(i));
                    nodes.push_back((float)spec.weight(j ,i, n,m));
                    nodes.push_back((float)std::max((float)spec.weight(j ,i, n,m),1.0f) / (float)std::max((float)spec.capacity(j, n), 1.0f));
                    nodes.push_back((float)spec.profit(i) / (float)std::max((float)spec.weight(j ,i, n,m),1.0f));
                    nodes.push_back(1.0f*i);
                    nodes.push_back(1.0f*j);
                }
            }

            at::Tensor nodes_t = torch::from_blob(nodes.data(), {m * n , 6}).to(torch::kFloat32);

            if (use_gpu)
                inputs.push_back(nodes_t.to(device));
            else
                inputs.push_back(nodes_t);

            // Prepare edges
            size_t num_edges = m * n * (n - 1) / 2 + m * n;
            std::vector<std::vector<int64_t>> edges_indexes_vec(2, std::vector<int64_t>(num_edges));
            std::vector<std::vector<int64_t>> edges_attr_vec(2, std::vector<int64_t>(num_edges));
            std::vector<float> edges_weights_vec(num_edges);

            int compteur = 0;
            for (int k = 0; k < m; k++) {
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        edges_indexes_vec[0][compteur] = k * n + i;
                        edges_indexes_vec[1][compteur] = k * n + j;
                        edges_weights_vec[compteur] = 1.0f / (float)n;
                        compteur++;
                    }
                }
            }

            for (int k = 0; k < n; k++) {
                for (int i = 0; i < m; i++) {
                    edges_indexes_vec[0][compteur] = i * n + k;
                    edges_indexes_vec[1][compteur] = k;
                    edges_weights_vec[compteur] = 1;
                    compteur++;
                }
            }

            // print the elements of edges_indexes_vec.data()
            at::Tensor edge_first = torch::from_blob(edges_indexes_vec[0].data(), {m * n * (n -1) / 2 + m * n}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edge_second = torch::from_blob(edges_indexes_vec[1].data(), {m * n * (n -1) / 2 + m * n}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, m * n * (n -1) / 2 + m * n});
            at::Tensor edges_weights = torch::from_blob(edges_weights_vec.data(), {m * n * (n -1) / 2 + m * n}, torch::TensorOptions().dtype(torch::kFloat32));

            if (use_gpu){
                inputs.push_back(edges_indexes.to(device));
                inputs.push_back(edges_weights.to(device));
            }
            else{
                inputs.push_back(edges_indexes);
                inputs.push_back(edges_weights);
            }   


            at::Tensor intermediate_output = module_1.forward(inputs).toTensor();
            at::Tensor mean = intermediate_output.mean(0).repeat({m * n, 1});

            std::vector<torch::jit::IValue> intermediate_inputs;
            at::Tensor intermediate_input_t = torch::cat({intermediate_output, mean}, 1);
            intermediate_inputs.push_back(intermediate_input_t.to(torch::kFloat32));
            at::Tensor multiplier_t = module_2.forward(intermediate_inputs).toTensor();

            // create a vector of multipliers
            std::vector<float> multipliers_vec;
            multipliers_vec.resize(m * n);
            for (int i = 0; i < m * n; i++) {
                multipliers_vec[i] = multiplier_t[i].item<float>();
            }

            this->multipliers.resize(n);
            for (int i = 0; i < n; ++i) {
                float sum = 0;
                this->multipliers[i].resize(m);
                for (int j = 1; j < m; ++j) {
                    this->multipliers[i][j] =  (float)multipliers_vec[i + j *n];
                    sum += (float)multipliers_vec[i + j *n];
                }
                this->multipliers[i][0] = sum;
            }
        }
        else{
            this->multipliers.resize(n);
            for (int i = 0; i < n; ++i) {
                float sum = 0;
                this->multipliers[i].resize(m);
                for (int j = 1; j < m; ++j) {
                    this->multipliers[i][j] = init_value_multipliers;
                    sum += init_value_multipliers;
                }
                this->multipliers[i][0] = sum;
            }
        }

        // Get the profits, capacities and weights of the items from the instance
        for (int i = 0; i < n; i++) {
            profits[i] = spec.profit(i);
        }

        for (int i = 0; i < m; i++) {
            capacities[i] = spec.capacity(i, n);
        }

        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                int w = spec.weight(j, i, n, m);
                weights[j][i] = w;
            }
        }

        this->order_branching.resize(n);

        // order is the list of the index of items sorted by decreasing ratio between profit and weight
        for (int i = 0; i < n; i++) {
          order_branching[i] = i;
        }

        std::vector<std::vector<float>> weights_f(m, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                weights_f[j][i] = weights[j][i];
            }
        }

        std::sort(order_branching.begin(), order_branching.end(), [&](int i, int j) { 
            float sum_i = 0;
            float sum_j = 0;
            for (int k = 0; k < m; k++) {
                sum_i += weights_f[k][i];
                sum_j += weights_f[k][j];
            }
            float ratio_i = profits[i] / std::max(sum_i, 1.0f);
            float ratio_j = profits[j] / std::max(sum_j, 1.0f);
            return ratio_i > ratio_j;
        });

        // The objective function
        IntVarArgs profits_x;
        for (int i = 0; i < n; i++) {
            profits_x << expr(*this, profits[i] * variables[i]);
        }
        z = expr(*this, sum(profits_x));


        // The constraints for the knapsacks
        for (int j = 0; j < m; j++) {
            IntVarArgs weight_x;
            for (int i = 0; i < n; i++) {
                weight_x << expr(*this, weights[j][i] * variables[i]);
            }
            linear(*this, weight_x, IRT_LQ, capacities[j]);
        }

        nonemax(*this, variables);
    }

    void compare(const Space& s, std::ostream& os) const {
    }

    void more(void) { // compute the bound at each node after every branching
        int nb_items = spec.nb_items();
        int nb_constraints = spec.nb_constraints();
        int capacity = spec.capacity(0, nb_items);
        float final_fixed_bounds = 0.0f;
        float beta1 = 0.9;
        float beta2 = 0.99;
        float epsilon = 1e-4;
        use_gpu = true;

        int rows = nb_items;
        int cols = nb_constraints;

        // store the value of the variable in the solution during the dynamic programming algo to update the multipliers
        int** value_var_solution = new int*[rows];

        // init value_var_solution with 0
        for (int i = 0; i < rows; ++i) {
            value_var_solution[i] = new int[cols];
            for (int j = 0; j < cols; ++j) {
                value_var_solution[i][j] = 0;
            }
        }

        int* diff_var_solution =new int[rows-1]; // store the difference between the value of the variable and the value of the other variables in the solution

        float final_bound = std::numeric_limits<float>::max();
        std::vector<float> bound_test;
        
        std::vector<int> not_fixed_variables;
        std::vector<int> fixed_variables;

        std::string dicstr(nb_items,' ');

        for (int k = 0; k < nb_items; k++){
            if (variables[k].size()==2) {
                dicstr[k]='2';
                not_fixed_variables.push_back(k);
                for (int j=0;j<cols;j++) {
                    value_var_solution[k][j]=0; }
            }

            else{
                fixed_variables.push_back(k);
                if (variables[k].val()==1) { 
                    dicstr[k]='1';
                    for (int j=0;j<cols;j++) {
                        value_var_solution[k][j]=variables[k].val(); }
                }
                else {
                    dicstr[k]='0';
                }
            }
        }

        int size_unfixed=not_fixed_variables.size();
        int* node_problem=new int[size_unfixed*(nb_constraints+1)+nb_constraints+2 + 1 + 1];

        node_problem[0]=nb_constraints;
        node_problem[1]=size_unfixed;

        for (int k=0;k<size_unfixed;k++) {
            node_problem[2+k]=spec.profit(not_fixed_variables[k]);
        }

        for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
            int capacity_unfixed=spec.capacity(idx_constraint,nb_items);

            for (int i=0;i<fixed_variables.size();i++) {
                capacity_unfixed-=spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints)*variables[fixed_variables[i]].val();
            }

            node_problem[2+size_unfixed+idx_constraint]=capacity_unfixed;
        }

        for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
            for (int i=0;i<size_unfixed;i++) {
                node_problem[2+(idx_constraint+1)*size_unfixed+nb_constraints+i]=spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);
            }
        }
        if (activate_learning_and_grad){
            std::vector<torch::jit::IValue> inputs;

            torch::Device device(torch::kCUDA);

            if (use_gpu)
                device = torch::Device(torch::kCUDA);


            std::vector<float> nodes;
            nodes.reserve(nb_constraints * size_unfixed * 6);

            for (int j=0;j<nb_constraints;j++) {
                for (int i=0;i<size_unfixed;i++) {
                    nodes.push_back(node_problem[2+i]);
                    nodes.push_back(node_problem[2+ size_unfixed  + nb_constraints + size_unfixed*j +i]);
                    nodes.push_back((float)std::max((float)node_problem[2+ + size_unfixed + + nb_constraints + size_unfixed*j +i],1.0f) / (float)std::max((float)node_problem[2 + size_unfixed + j], 1.0f));
                    nodes.push_back((float)node_problem[2 + i] / (float)std::max((float)node_problem[2+ + size_unfixed + nb_constraints +size_unfixed*j +i],1.0f));
                    nodes.push_back(i);
                    nodes.push_back(j);
                }
            }

            at::Tensor nodes_t = torch::from_blob(nodes.data(), {nb_constraints * size_unfixed , 6}).to(torch::kFloat32);
            
            if (use_gpu)
                inputs.push_back(nodes_t.to(device));
            else
                inputs.push_back(nodes_t);

            std::vector<std::vector<int64_t>> edges_indexes_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
            std::vector<std::vector<int64_t>> edges_attr_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
            std::vector<float> edges_weights_vec(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed);

            int compteur = 0;
            for (int k = 0; k < nb_constraints; k++) {
                for (int i = 0; i < size_unfixed; i++) {
                    for (int j = i + 1; j < size_unfixed; j++) {
                        edges_indexes_vec[0][compteur] = k * size_unfixed + i;
                        edges_indexes_vec[1][compteur] = k * size_unfixed + j;
                        edges_weights_vec[compteur] = 1.0f / (float)size_unfixed;
                        compteur++;
                    }
                }
            }

            for (int k = 0; k < size_unfixed; k++) {
                for (int i = 0; i < nb_constraints; i++) {
                    edges_indexes_vec[0][compteur] = i * size_unfixed + k;
                    edges_indexes_vec[1][compteur] = k;
                    edges_weights_vec[compteur] = 1;
                    compteur++;
                }
            }

            // print the elements of edges_indexes_vec.data()
            at::Tensor edge_first = torch::from_blob(edges_indexes_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edge_second = torch::from_blob(edges_indexes_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
            at::Tensor edges_weights = torch::from_blob(edges_weights_vec.data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kFloat32));

            if (use_gpu){
                inputs.push_back(edges_indexes.to(device));
                inputs.push_back(edges_weights.to(device));
            }
            else{
                inputs.push_back(edges_indexes);
                inputs.push_back(edges_weights);
            } 

            at::Tensor intermediate_output = module_1.forward(inputs).toTensor();

            at::Tensor mean = intermediate_output.mean(0).repeat({nb_constraints * size_unfixed, 1});

            std::vector<torch::jit::IValue> intermediate_inputs;

            intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));
            at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

            std::vector<std::vector<float>> multipliers_vec(rows, std::vector<float>(cols, 0.0));
            for (int i = 0; i < size_unfixed; ++i) {
                for (int j = 0; j < cols; ++j) {
                    multipliers_vec[not_fixed_variables[i]][j] = multipliers[i + j*size_unfixed].item<float>();
                }
            }
          
            learning_rate = 1.0f;

            int nb_iter = 1;
            while ((( nb_iter < 5) || (abs(bound_test[nb_iter-2] - bound_test[nb_iter-3]) / bound_test[nb_iter-2] > 1e-6))&& (nb_iter < 6)) { // We repeat the dynamic programming algo to solve the knapsack problem
                                      // and at each iteration we update the value of the Lagrangian multipliers
                final_fixed_bounds = 0.0f;
                float bound_iter = 0.0f;
                std::vector<SubProblem> subproblems;

                for (int i = 0; i < size_unfixed; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        value_var_solution[not_fixed_variables[i]][j] = 0;
                    }
                }

                // we create one subproblem for each knapsack constraint
                for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
                    SubProblem subproblem;
                    subproblem.weights_sub = new int[nb_items];
                    subproblem.val_sub = new float[nb_items];
                    subproblem.capacity = spec.capacity(idx_constraint, nb_items);

                    for (int i = 0; i < fixed_variables.size(); i++) {
                        subproblem.weights_sub[i] = spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints);

                        if (idx_constraint == 0) {
                            subproblem.val_sub[i] = spec.profit(fixed_variables[i]);
                        }

                        else {
                            subproblem.val_sub[i] = 0.0f;
                        }
                    }
                    for (int i = 0; i < not_fixed_variables.size(); i++) {
                        subproblem.weights_sub[i + fixed_variables.size()] = spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);

                        if (idx_constraint == 0) {
                            float mult = 0.0f;
                            for (int j = 1; j < nb_constraints; j++) {
                                mult += multipliers_vec[not_fixed_variables[i]][j];
                            }
                            subproblem.val_sub[i + fixed_variables.size()] = spec.profit(not_fixed_variables[i]) + mult;
                        }
                        else {
                            subproblem.val_sub[i + fixed_variables.size()] = - multipliers_vec[not_fixed_variables[i]][idx_constraint];
                        }
                    }
                    subproblem.idx_constraint = idx_constraint;
                    subproblems.push_back(subproblem);
                }

                for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
                    SubProblem subproblem = subproblems[id_subproblem];
                    float weight_fixed = 0.0f;

                    for (int k = 0; k < fixed_variables.size(); k++){
                        weight_fixed+=subproblem.weights_sub[k] * variables[fixed_variables[k]].val();
                    }

                    float bound = dp_knapsack(subproblem.capacity - weight_fixed,
                                              subproblem.weights_sub + fixed_variables.size(),
                                              subproblem.val_sub + fixed_variables.size(),
                                              nb_items - fixed_variables.size(), nb_constraints,
                                              subproblem.idx_constraint,
                                              value_var_solution,
                                              not_fixed_variables,
                                              false);

                    float fixed_bound = 0.0f;

                    for (int k = 0; k< fixed_variables.size();k++){
                        fixed_bound += subproblem.val_sub[k] * variables[fixed_variables[k]].val();
                    }
                    final_fixed_bounds += fixed_bound;

                    bound_iter += bound + fixed_bound; // sum all the bound of the knapsack sub-problem to update the multipliers
                
                }
                final_bound = std::min(final_bound, bound_iter);
                bound_test.push_back(final_bound);


                for (int i = 0; i < rows; ++i) {
                    float sum = 0.0f;
                    for (int j = 1; j < cols; ++j) {

                        float gradient = value_var_solution[i][0] - value_var_solution[i][j];

                        multipliers_vec[i][j] -= 1 * gradient;

                        sum += multipliers_vec[i][j];
                    }
                    multipliers_vec[i][0] = sum;
                }

                // We impose the constraint z <= final_bound
                nb_iter++;
                for (auto& subproblem : subproblems) {
                    delete[] subproblem.weights_sub;
                    delete[] subproblem.val_sub;
                }
            }

            rel(*this, z <= std::ceil(final_bound)); 
            compteur_iterations += nb_iter-1;
        }
        else if (activate_learning_prediction){
            std::vector<torch::jit::IValue> inputs;
            torch::Device device(torch::kCPU);

            if (use_gpu)
                device = torch::Device(torch::kCUDA);

            std::vector<float> nodes;
            nodes.reserve(nb_constraints * size_unfixed * 6);

            for (int j=0;j<nb_constraints;j++) {
                for (int i=0;i<size_unfixed;i++) {
                    nodes.push_back(node_problem[2+i]);
                    nodes.push_back(node_problem[2+ size_unfixed  + nb_constraints + size_unfixed*j +i]);
                    nodes.push_back((float)std::max((float)node_problem[2+ + size_unfixed + + nb_constraints + size_unfixed*j +i],1.0f) / (float)std::max((float)node_problem[2 + size_unfixed + j], 1.0f));
                    nodes.push_back((float)node_problem[2 + i] / (float)std::max((float)node_problem[2+ + size_unfixed + nb_constraints +size_unfixed*j +i],1.0f));
                    nodes.push_back(i);
                    nodes.push_back(j);
                }
            }

            at::Tensor nodes_t = torch::from_blob(nodes.data(), {nb_constraints * size_unfixed , 6}).to(torch::kFloat32);
              
            if (use_gpu)
                inputs.push_back(nodes_t.to(device));
            else
                inputs.push_back(nodes_t);

            std::vector<std::vector<int64_t>> edges_indexes_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
            std::vector<std::vector<int64_t>> edges_attr_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
            std::vector<float> edges_weights_vec(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed);

            int compteur = 0;
            for (int k = 0; k < nb_constraints; k++) {
                for (int i = 0; i < size_unfixed; i++) {
                    for (int j = i + 1; j < size_unfixed; j++) {
                        edges_indexes_vec[0][compteur] = k * size_unfixed + i;
                        edges_indexes_vec[1][compteur] = k * size_unfixed + j;
                        edges_weights_vec[compteur] = 1.0f / (float)size_unfixed;
                        compteur++;
                    }
                }
            }
            for (int k = 0; k < size_unfixed; k++) {
                for (int i = 0; i < nb_constraints; i++) {
                    edges_indexes_vec[0][compteur] = i * size_unfixed + k;
                    edges_indexes_vec[1][compteur] = k;
                    edges_weights_vec[compteur] = 1;
                    compteur++;
                }
            }
            // print the elements of edges_indexes_vec.data()
            at::Tensor edge_first = torch::from_blob(edges_indexes_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edge_second = torch::from_blob(edges_indexes_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
            at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
            at::Tensor edges_weights = torch::from_blob(edges_weights_vec.data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kFloat32));

            if (use_gpu){
                inputs.push_back(edges_indexes.to(device));
                inputs.push_back(edges_weights.to(device));
            }
            else{
                inputs.push_back(edges_indexes);
                inputs.push_back(edges_weights);
            } 
            at::Tensor intermediate_output = module_1.forward(inputs).toTensor();

            at::Tensor mean = intermediate_output.mean(0).repeat({nb_constraints * size_unfixed, 1});

            std::vector<torch::jit::IValue> intermediate_inputs;

            intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));
            at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

            // create a vector of multipliers
            std::vector<float> multipliers_vec;
            multipliers_vec.resize(nb_constraints * size_unfixed);
            for (int i = 0; i < nb_constraints * size_unfixed; i++) {
                multipliers_vec[i] = multipliers[i].item<float>();
            }

            float final_bound = 0.0f;
            std::vector<SubProblem> subproblems;

                // we create one subproblem for each knapsack constraint
            for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
                SubProblem subproblem;
                subproblem.weights_sub = new int[nb_items];
                subproblem.val_sub = new float[nb_items];
                subproblem.capacity = spec.capacity(idx_constraint, nb_items);

                for (int i = 0; i < fixed_variables.size(); i++) {
                    subproblem.weights_sub[i] = spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints);

                    if (idx_constraint == 0) {
                        subproblem.val_sub[i] = spec.profit(fixed_variables[i]);
                    }

                    else {
                        subproblem.val_sub[i] = 0.0f;
                    }
                 }
                for (int i = 0; i < not_fixed_variables.size(); i++) {
                    subproblem.weights_sub[i + fixed_variables.size()] = spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);

                    if (idx_constraint == 0) {
                        float mult = 0.0f;
                        for (int j = 1; j < nb_constraints; j++) {
                            mult += multipliers_vec[i + size_unfixed * j];
                        }
                        subproblem.val_sub[i + fixed_variables.size()] = spec.profit(not_fixed_variables[i]) + mult ;
                    }

                    else {
                        subproblem.val_sub[i + fixed_variables.size()] = - multipliers_vec[i + size_unfixed * idx_constraint];
                    }
                }
                subproblem.idx_constraint = idx_constraint;
                subproblems.push_back(subproblem);
            } 

            for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
                SubProblem subproblem = subproblems[id_subproblem];
                float weight_fixed = 0.0f;
            
                for (int k = 0; k < fixed_variables.size(); k++){
                    weight_fixed+=subproblem.weights_sub[k]* variables[fixed_variables[k]].val();
                }

                float bound = dp_knapsack(subproblem.capacity - weight_fixed,
                                            subproblem.weights_sub + fixed_variables.size(),
                                            subproblem.val_sub + fixed_variables.size(),
                                            nb_items - fixed_variables.size(), nb_constraints,
                                            subproblem.idx_constraint,
                                            value_var_solution,
                                            not_fixed_variables,
                                            false);

                float fixed_bound = 0.0f;

                for (int k = 0; k< fixed_variables.size();k++){
                    fixed_bound += subproblem.val_sub[k] * variables[fixed_variables[k]].val();
                }
                final_fixed_bounds += fixed_bound;
                final_bound += bound + fixed_bound; // sum all the bound of the knapsack sub-problem to update the multipliers
            }
            for (auto& subproblem : subproblems) {
                  delete[] subproblem.weights_sub;
                  delete[] subproblem.val_sub;
            }

            // We impose the constraint z <= final_bound
            rel(*this, z <= std::ceil(final_bound)); 
        }
        else{
            int nb_iter = 1;
            while (( nb_iter < 59) || (abs(bound_test[nb_iter-2] - bound_test[nb_iter-3]) / bound_test[nb_iter-2] > 1e-6) && (nb_iter < 60)) { 
                // We repeat the dynamic programming algo to solve the knapsack problem
                // and at each iteration we update the value of the Lagrangian multipliers
                final_fixed_bounds = 0.0f;
                float bound_iter = 0.0f;
                std::vector<SubProblem> subproblems;

                for (int i = 0; i < size_unfixed; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        value_var_solution[not_fixed_variables[i]][j] = 0;
                    }
                }

                // we create one subproblem for each knapsack constraint
                for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
                    SubProblem subproblem;
                    subproblem.weights_sub = new int[nb_items];
                    subproblem.val_sub = new float[nb_items];
                    subproblem.capacity = spec.capacity(idx_constraint, nb_items);

                    for (int i = 0; i < fixed_variables.size(); i++) {
                        subproblem.weights_sub[i] = spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints);

                        if (idx_constraint == 0) {
                            subproblem.val_sub[i] = spec.profit(fixed_variables[i]) + multipliers[fixed_variables[i]][idx_constraint];
                        }

                        else {
                            subproblem.val_sub[i] = - multipliers[fixed_variables[i]][idx_constraint];
                        }
                    }
                    for (int i = 0; i < not_fixed_variables.size(); i++) {
                        subproblem.weights_sub[i + fixed_variables.size()] = spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);

                        if (idx_constraint == 0) {
                            subproblem.val_sub[i + fixed_variables.size()] = spec.profit(not_fixed_variables[i]) + multipliers[not_fixed_variables[i]][idx_constraint];
                        }

                        else {
                            subproblem.val_sub[i + fixed_variables.size()] = - multipliers[not_fixed_variables[i]][idx_constraint];
                        }
                    }
                    subproblem.idx_constraint = idx_constraint;
                    subproblems.push_back(subproblem);
                }

                for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
                    SubProblem subproblem = subproblems[id_subproblem];
                    float weight_fixed = 0.0f;
            
                    for (int k = 0; k < fixed_variables.size(); k++){
                        weight_fixed+=subproblem.weights_sub[k]* variables[fixed_variables[k]].val();
                    }

                    float bound = dp_knapsack(subproblem.capacity - weight_fixed,
                                          subproblem.weights_sub + fixed_variables.size(),
                                          subproblem.val_sub + fixed_variables.size(),
                                          nb_items - fixed_variables.size(), nb_constraints,
                                          subproblem.idx_constraint,
                                          value_var_solution,
                                          not_fixed_variables,
                                          false); 

                    float fixed_bound = 0.0f;

                    for (int k = 0; k< fixed_variables.size();k++){
                        fixed_bound += subproblem.val_sub[k] * variables[fixed_variables[k]].val();
                    }
                    final_fixed_bounds += fixed_bound;

                    bound_iter += bound + fixed_bound; // sum all the bound of the knapsack sub-problem to update the multipliers
            
                }
                final_bound = std::min(final_bound, bound_iter);
                bound_test.push_back(bound_iter);

                for (int i = 0; i < rows; ++i) {
                    float sum = 0;
                    for (int j = 1; j < cols; ++j) {

                        multipliers[i][j] = multipliers[i][j] -  learning_rate *  (value_var_solution[i][0] - value_var_solution[i][j]);

                        sum += multipliers[i][j];
                    }
                    multipliers[i][0] = sum;
                }

                for (auto& subproblem : subproblems) {
                    delete[] subproblem.weights_sub;
                    delete[] subproblem.val_sub;
                }

                // We impose the constraint z <= final_bound
                rel(*this, z <= std::ceil(final_bound)); 
                nb_iter++;
            }
            compteur_iterations += nb_iter-1;
        }

        node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+2] = final_fixed_bounds;
        node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+2 + 1] = final_bound;

        if (write_samples) {
            if ((set_nodes.count(dicstr)==0) and (size_unfixed>=3 )) { // we write the node only if it is not already in the set
                set_nodes.insert(dicstr);

                if (outputFileMK->is_open()) {
                    for (int i=0;i<size_unfixed*(nb_constraints+1)+nb_constraints+3;i++) {
                        *outputFileMK << node_problem[i]<<",";
                      }

                    *outputFileMK << node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+3]<<"\n";
                } 
            } 
        }

        for (int i = 0; i < rows; ++i) {
            delete[] value_var_solution[i];
        }

        delete[] value_var_solution;

        delete[] node_problem;
    }

    static void post(Space& home) {
        static_cast<MultiKnapsack&>(home).more();
    }

    /// Return cost
    virtual IntVar cost(void) const {
        return z;
    }

    /// Constructor for cloning \a s
    MultiKnapsack(MultiKnapsack& s)
        : IntMaximizeSpace(s), spec(s.spec) {
          variables.update(*this, s.variables);
          z.update(*this, s.z);
          this->order_branching = s.order_branching;
          this->activate_bound_computation = s.activate_bound_computation;
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
    virtual Space* copy(void) {
        return new MultiKnapsack(*this);
    }

    virtual void constrain(const Space& _b) { // compute the bound at each leaf node giving a solution
      const MultiKnapsack& b = static_cast<const MultiKnapsack&>(_b);
      std::cout<<"solution: "<<b.z<<std::endl;
      // We impose the constraint z >= current sol
      rel(*this, z >= b.z);
  }

    float dp_knapsack(int capacity,
                      int weights[],
                      float val[],
                      int nb_items,
                      int nb_constraints,
                      int idx_constraint,
                      int** value_var_solution,
                      std::vector<int>& not_fixed_variables,
                      bool verbose=false) {
        std::vector<std::vector<float>> dp(capacity + 1, std::vector<float>(nb_items + 1, 0.0f));
        for (int i = 1; i <= nb_items; ++i) {
            for (int w = 0; w <= capacity; ++w) {
                if (weights[i - 1] <= w) {
                    dp[w][i] = std::max(dp[w][i - 1], dp[w - weights[i - 1]][i - 1] + val[i - 1]);
                } 
                else {
                    dp[w][i] = dp[w][i - 1];
                }
            }
        }
        // Backtracking to find selected items
        int w = capacity;
        for (int i = nb_items; i > 0; --i) {
            if (dp[w][i] != dp[w][i - 1]) {
                value_var_solution[not_fixed_variables[i - 1]][idx_constraint] = 1;
                w -= weights[i - 1];
            }
        }
        return dp[capacity][nb_items];
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
    int number_of_sizes = std::stoi(n_size);
    int number_of_models = std::stoi(n_model);
    int start_size = std::stoi(size_start);
    int start_model = std::stoi(model_start);

    bool write_samples;
    bool pissinger;
    bool use_gpu;

    if (strcmp(argv[5], "write_samples") == 0)
        write_samples = true;
    else
        write_samples = false;

    if (strcmp(argv[6], "pissinger") == 0)
        pissinger = true;
    else
        pissinger = false;

    if (strcmp(argv[7], "gpu") == 0)
        use_gpu = true;
    else
        use_gpu = false;

    std::string n_file = argv[8];

    std::string sizes[] = {"30", "50", "100", "200"};

    bool activate_bound_computation[] = {true, true, true, true, false};
    bool activate_learning_prediction[] = {false, false, true, false, false};
    bool activate_learning_and_grad[] = {false, false, false, true, false};
    bool activate_init_learning[] = {false, true, false, true, false};
    bool activate_heuristic[] = {true, true, true, true, true};
    int K = 500;
    float learning_rate = 1.0f;
    float init_value_multipliers = 1.0f;

    std::cout<<"beginning of the program"<<std::endl;

    if (write_samples) {
        if (pissinger) {
            for (int index_size = start_size; index_size < number_of_sizes; index_size++) {
        
                std::ifstream inputFilea("../../../../data/mknapsack/train/pissinger-unordered/knapsacks-data-trainset" + sizes[index_size] + ".txt");
                bool activate_bound_computation = true;
                bool activate_learning_prediction = false;
                bool activate_learning_and_grad = false;
                bool activate_init_learning = false;
                bool activate_heuristic = true;

                std::string line;
                std::vector<int> problem;
                std::vector<int> numbers;
                int j = 233;

                while (std::getline(inputFilea, line)) {

                    std::ofstream outputFilea("../../../../data/mknapsack/train/pissinger-unordered/trainset-mk"+ sizes[index_size]+"-subnodes" + std::to_string(j) +  ".txt");


                    set_nodes.clear();
                    std::vector<int> problem;
                    std::istringstream iss(line);
                    std::string substring;
                    while (std::getline(iss, substring, ',')) {
                        problem.push_back(std::stoi(substring));
                    }
                    std::cout<<""<<std::endl;

                    OptionsKnapsack opt=OptionsKnapsack(activate_bound_computation, activate_init_learning, activate_learning_prediction, activate_learning_and_grad, activate_heuristic, use_gpu, K,learning_rate,init_value_multipliers, &outputFilea , problem, true);
                        opt.instance();
                        opt.solutions(0);
                        opt.parse(argc, argv);
                        IntMaximizeScript::run<MultiKnapsack,BAB,OptionsKnapsack>(opt);
                    
                    j++;
                    outputFilea.close();
                }
                inputFilea.close(); // Close the file when done
            }           

        }
        else {
            std::ifstream inputFilea("../../../../data/mknapsack/train/weish/knapsacks-data.txt");
            bool activate_bound_computation = true;
            bool activate_learning_prediction = false;
            bool activate_learning_and_grad = false;
            bool activate_init_learning = false;
            bool activate_heuristic = true;

            std::string line;
            std::vector<int> problem;
            std::vector<int> numbers;
            int j = 1;
            while (std::getline(inputFilea, line)) {

                try{
                    std::ofstream outputFilea("../../../../data/mknapsack/train/weish/weish-train-mk-subnodes" + std::to_string(j) +  ".txt");

                    set_nodes.clear();
                    std::vector<int> problem;
                    std::istringstream iss(line);
                    std::string substring;
                    while (std::getline(iss, substring, ',')) {
                        problem.push_back(std::stoi(substring));
                    }
                        
                    std::cout<<""<<std::endl;

                    OptionsKnapsack opt=OptionsKnapsack(activate_bound_computation, activate_init_learning, activate_learning_prediction, activate_learning_and_grad, activate_heuristic, use_gpu, K,learning_rate,init_value_multipliers, &outputFilea , problem, true);
                        opt.instance();
                        opt.solutions(0);
                        opt.parse(argc, argv);
                        IntMaximizeScript::run<MultiKnapsack,BAB,OptionsKnapsack>(opt);
                    j++;
                    outputFilea.close();
                }
                catch (const std::invalid_argument& ia) { 
                    std::cout << "error with the instance" << std::endl;
                }
            }
            inputFilea.close(); // Close the file when done
        }
    }

    else {
        if (pissinger) {
            for (int index_size = start_size; index_size < number_of_sizes; index_size++) {

                std::cout<<"separateur de modeles" << std::endl;
                for (int index_model = start_model; index_model < number_of_models ; index_model++ ){
                    try {
                        if (use_gpu) {
                            // Deserialize the ScriptModule from a file using torch::jit::load().
                            if (activate_init_learning[index_model]){
                                module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-GPU" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-GPU"+ sizes[index_size]+ ".pt");
                            }
                            else{
                                module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-GPU" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-GPU"+ sizes[index_size]+ ".pt");
                            }
                        }
                        else {
                            if (activate_init_learning[index_model]){
                                module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-CPU-root-" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-CPU-root-"+ sizes[index_size]+ ".pt");
                            } 
                            else{
                                module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-CPU-root-" + sizes[index_size] + ".pt");
                                module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-CPU-root-"+ sizes[index_size]+ ".pt");
                            }                   
                        }
                     }
                    catch (const c10::Error& e) {
                        std::cerr << "error with loading the models \n";
                        // return -1;
                    }

                    for (int i = 0; i < 1; i++) {
                        std::ifstream inputFilea("../../../../data/mknapsack/test/pissinger/knapsacks-data-testset" + sizes[index_size] + + "-" + n_file + ".txt");
                        std::string line;
                        std::vector<int> problem;
                        std::vector<int> numbers;
                        int j =1;

                        while (std::getline(inputFilea, line)) {
                            compteur_iterations = 0;
                            std::vector<int> problem;
                            std::istringstream iss(line);
                            std::string substring;
                                while (std::getline(iss, substring, ',')) {
                                    problem.push_back(std::stoi(substring));
                                }
                            std::cout<<""<<std::endl;

                            OptionsKnapsack opt=OptionsKnapsack(activate_bound_computation[index_model], activate_init_learning[index_model], activate_learning_prediction[index_model], activate_learning_and_grad[index_model], activate_heuristic[index_model], use_gpu, K,learning_rate,init_value_multipliers, NULL , problem, false);
                            opt.instance();
                            opt.solutions(0);
                            opt.parse(argc, argv);
                            IntMaximizeScript::run<MultiKnapsack,BAB,OptionsKnapsack>(opt);
                            std::cout << "compteur_iterations : " << compteur_iterations << std::endl;
                        }
                  
                        std::cout<<"separateur_de_modeles"<<std::endl;
                        inputFilea.close(); // Close the file when done
                    }
                }
            std::cout << "separateur de taille" << std::endl;
            }
        }
        else{
            std::cout<<"separateur de modeles" << std::endl;
            for (int index_model = start_model; index_model < number_of_models ; index_model++ ){
                try {
                    if (use_gpu) {
                        // Deserialize the ScriptModule from a file using torch::jit::load().
                        if (activate_init_learning[index_model]){
                            module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-weish-GPU.pt");
                            module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-weish-GPU.pt");
                        }
                        else{
                            module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-weish-GPU.pt");
                            module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-weish-GPU.pt");
                        }
                    }
                    else {
                        if (activate_init_learning[index_model]){
                            module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-weish-CPU.pt");
                            module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-weish-CPU.pt");
                        }
                        else{
                            module_1 = torch::jit::load("../../../../trained_models/mknapsack/model_graph_representation-weish-CPU.pt");
                            module_2 = torch::jit::load("../../../../trained_models/mknapsack/model_prediction-weish-CPU.pt");
                        }                   
                    }
                }
                catch (const c10::Error& e) {
                    std::cerr << "error with loading the models \n";
                    return -1;
               }

                for (int i = 0; i < 1; i++) {
                    std::ifstream inputFilea("../../../../data/mknapsack/test/weish/knapsacks-data.txt");
                    std::string line;
                    std::vector<int> problem;
                    std::vector<int> numbers;
                    int j =1;

                    while (std::getline(inputFilea, line)) {
                        compteur_iterations = 0;
                        std::vector<int> problem;
                        std::istringstream iss(line);
                        std::string substring;
                            while (std::getline(iss, substring, ',')) {
                                problem.push_back(std::stoi(substring));
                            }
                        std::cout<<""<<std::endl;
                        OptionsKnapsack opt=OptionsKnapsack(activate_bound_computation[index_model],  activate_init_learning[index_model], activate_learning_prediction[index_model], activate_learning_and_grad[index_model], activate_heuristic[index_model], use_gpu, K,learning_rate,init_value_multipliers, NULL , problem, false);
                            opt.instance();
                            opt.solutions(0);
                            opt.parse(argc, argv);
                            IntMaximizeScript::run<MultiKnapsack,BAB,OptionsKnapsack>(opt);
                        std::cout << "compteur_iterations : " << compteur_iterations << std::endl;
                    }
              
                    std::cout<<"separateur_de_modeles"<<std::endl;
                    inputFilea.close(); // Close the file when done
                }
            }
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
