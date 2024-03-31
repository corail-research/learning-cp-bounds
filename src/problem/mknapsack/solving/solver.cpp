#include <gecode/driver.hh>

#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include <algorithm>

using namespace Gecode;

// Allow to store the multipliers in a local handle to share them between the nodes
class LI : public LocalHandle {
protected:
  class LIO : public LocalObject {
  public:
    std::vector<std::vector<float>> multipliers;
    LIO(Space& home, std::vector<std::vector<float>> d) : LocalObject(home), multipliers(d) {}
    LIO(Space& home, LIO& l)
      : LocalObject(home,l), multipliers(l.multipliers) {}
    virtual LocalObject* copy(Space& home) {
      return new (home) LIO(home,*this);
    }
    virtual size_t dispose(Space&) { return sizeof(*this); }
  };
public:
  LI(Space& home, std::vector<std::vector<float>> d)
    : LocalHandle(new (home) LIO(home,d)) {}
  LI(const LI& li)
    : LocalHandle(li) {}
  LI& operator =(const LI& li) {
    return static_cast<LI&>(LocalHandle::operator =(li));
  }
  std::vector<std::vector<float>> get(void) const {
    return static_cast<LIO*>(object())->multipliers;
  }
  void set(std::vector<std::vector<float>> d) {
    static_cast<LIO*>(object())->multipliers = d;
  }
};

// Structure of the subproblems of the knapsack problem
struct SubProblem {
  int* weights_sub;
  float* val_sub;
  int capacity;
  int idx_constraint;
};

// Assign labels 0 and 1 to the edges in the reduced graph
struct LabelEdge {
  std::pair<int, int> pair1; // (weight, index item)
  std::pair<int, int> pair2; // (weight, index item)
  int label; // 0 or 1
};

// Assign a cost equals to the value of the objective function if the variables is set to 1, for the labels set to 1
struct CostEdge {
  std::pair<int, int> pair1; // (w, i)
  std::pair<int, int> pair2; // (w, i)
  float cost; // 0 or 1
};

// compute the sum of all the cost among all the paths in the reduced graph
struct Path {
  std::vector<CostEdge> path;
  float cost;
};

// Instance data
namespace {
  // Instances
  extern const int* mknps[];
  // Instance names
  extern const char* name[];

  /// A wrapper class for instance data
  class Spec {
  protected:
    /// Raw instance data
    const int* pData;
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
    static const int* find(const char* s) {
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

  public:
    /// Initialize
    Spec(const char* s) : pData(find(s)), l(0), u(0) {
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

// Entered by the user
class OptionsKnapsack: public InstanceOptions {
public:
  bool activate_bound_computation;
  int K;
  float learning_rate;
  float init_value_multipliers;
  OptionsKnapsack(const char* s, bool activate_bound_computation0, int K0, float learning_rate0, float init_value_multipliers0)
    : InstanceOptions(s), activate_bound_computation(activate_bound_computation0), K(K0), learning_rate(learning_rate0), init_value_multipliers(init_value_multipliers0) {}
};

// TODO : update the multipliers with the article method
class MultiKnapsack : public IntMaximizeScript {
protected:
    const Spec spec; // Specification of the instance
    BoolVarArray x; // Decision variables for each item
    IntVar z; // Variable for the objective function
    bool activate_bound_computation;
    int K;
    float learning_rate;
    float init_value_multipliers;
    std::vector<std::vector<float>> multipliers; // Lagrangian multipliers shared between the nodes
public:

  /// Actual model
  MultiKnapsack(const OptionsKnapsack& opt)
  : IntMaximizeScript(opt),
    spec(opt.instance()), 
    x(*this, spec.nb_items(), 0, 1),
    z(*this, spec.lower(), spec.upper()){
      
      int n = spec.nb_items();        // The number of items
      int m = spec.nb_constraints();  // The number of constraints
      int profits[n];                 // The profit of the items
      int capacities[m];              // The capacities of the knapsacks
      int weights[m][n];              // The weights of the items in the knapsacks (one vector per knapsack)  
      this->activate_bound_computation = opt.activate_bound_computation; // Activate the bound computation at each node
      this->K = opt.K;                // The number of iteration to find the optimal multipliers
      this->learning_rate = opt.learning_rate; // The learning rate to update the multipliers
      this->init_value_multipliers = opt.init_value_multipliers; // The starting value of the multipliers

      std::vector<std::vector<float>> v; // help to initialize the local handle which will contain the multipliers and be shared between the nodes
      v.resize(n);
      for (int i = 0; i < n; ++i) {
          float sum = 0;
          v[i].resize(m);
          for (int j = 1; j < m; ++j) {
              v[i][j] = init_value_multipliers;
              sum += init_value_multipliers;
          }
          v[i][0] = sum;
      }
      LI multipliers(*this, v);
      this->multipliers = multipliers.get();

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

      // The objective function
      IntVarArgs profits_x;
      for (int i = 0; i < n; i++) {
          profits_x << expr(*this, profits[i] * x[i]);
      }
      z = expr(*this, sum(profits_x));

      // The constraints for the knapsacks
      for (int j = 0; j < m; j++) {
          IntVarArgs weight_x;
          for (int i = 0; i < n; i++) {
              weight_x << expr(*this, weights[j][i] * x[i]);
          }
          linear(*this, weight_x, IRT_LQ, capacities[j]);          
      }

      if (activate_bound_computation) {
        for (int i = 0; i<x.size(); i++) {
          branch(*this, x[i], BOOL_VAL_MAX());
          branch(*this, &MultiKnapsack::post); // compute the bound at each node
        }
      }
      else {
        branch(*this, x, BOOL_VAR_NONE(), BOOL_VAL_MAX());
      }
  }

  void more(void) { // compute the bound at each node after every branching
    float copy_learning_rate = learning_rate;
    int nb_items = spec.nb_items();
    int nb_constraints = spec.nb_constraints();
    int capacity = spec.capacity(0, nb_items);
    int weights[nb_items];
    int val[nb_items];

    for (int i = 0; i < nb_items; i++) {
      weights[i] = spec.weight(0, i, nb_items, nb_constraints);
      val[i] = spec.profit(i);
    }

    int rows = nb_items;
    int cols = nb_constraints;

    // store the value of the variable in the solution during the dynamic programming algo to update the multipliers
    int** value_var_solution = new int*[rows];
    for (int i = 0; i < rows; ++i) {
        value_var_solution[i] = new int[cols];
    }

    float final_bound = 0.0f;
    float bound_test[K];

    for (int k=0; k<K; k++) { // We repeat the dynamic programming algo to solve the knapsack problem 
                              // and at each iteration we update the value of the Lagrangian multipliers
      float bound_iter = 0.0f;
      std::vector<SubProblem> subproblems;
      // we create one subproblem for each knapsack constraint
      for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
        SubProblem subproblem;
        subproblem.weights_sub = new int[nb_items];
        subproblem.val_sub = new float[nb_items];
        subproblem.capacity = spec.capacity(idx_constraint, nb_items);
        for (int i = 0; i < nb_items; i++) {
          subproblem.weights_sub[i] = spec.weight(idx_constraint, i, nb_items, nb_constraints);
          if (idx_constraint == 0) {
            subproblem.val_sub[i] = spec.profit(i) + multipliers[i][idx_constraint];
          }
          else {
            subproblem.val_sub[i] = -multipliers[i][idx_constraint];
          }        
        }
        subproblem.idx_constraint = idx_constraint;
        subproblems.push_back(subproblem);
      }

      for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
        SubProblem subproblem = subproblems[id_subproblem];
        float bound = dp_knapsack(subproblem.capacity, 
                                  subproblem.weights_sub, 
                                  subproblem.val_sub, 
                                  nb_items, nb_constraints, 
                                  subproblem.idx_constraint, 
                                  value_var_solution, 
                                  false);
        bound_iter += bound; // sum all the bound of the knapsack sub-problem to update the multipliers
      }
      final_bound = bound_iter;
      bound_test[k] = bound_iter;

      if (k >= 4) { // we divide by 2 the learning rate if the bound doesn't change 5 times in a row
        float b1 = bound_test[k];
        float b2 = bound_test[k-1];
        float b3 = bound_test[k-2];
        float b4 = bound_test[k-3];
        float b5 = bound_test[k-4];

        bool cond1 = (fabs(b1 - b2) <= 0.0001 * std::max(fabs(b1), fabs(b2)));
        bool cond2 = (fabs(b2 - b3) <= 0.0001 * std::max(fabs(b2), fabs(b3)));
        bool cond3 = (fabs(b3 - b4) <= 0.0001 * std::max(fabs(b3), fabs(b4)));
        bool cond4 = (fabs(b4 - b5) <= 0.0001 * std::max(fabs(b4), fabs(b5)));

        if (cond1 && cond2 && cond3 && cond4) {
          learning_rate = learning_rate/2;
        }
      }
  }

  // Update the multipliers (Quentin method with constant learning rate) TODO : implement the article method with adaptive learning rate
  for (int i = 0; i < rows; ++i) {
    float sum = 0;
    for (int j = 1; j < cols; ++j) {
      multipliers[i][j] = multipliers[i][j] - learning_rate * (value_var_solution[i][0] - value_var_solution[i][j]);
      sum += multipliers[i][j];
    }
    multipliers[i][0] = sum;
  }

  // We impose the constraint z <= final_bound
  rel(*this, z <= final_bound);

  learning_rate = copy_learning_rate;

  for (int i = 0; i < rows; ++i) {
    delete[] value_var_solution[i];
  }

  delete[] value_var_solution;  
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
    : IntMaximizeScript(s), spec(s.spec) {
    x.update(*this, s.x);
    z.update(*this, s.z);
    this->activate_bound_computation = s.activate_bound_computation;
    this->K = s.K;
    this->learning_rate = s.learning_rate;
    this->init_value_multipliers = s.init_value_multipliers;
    this->multipliers = s.multipliers;
  }
  /// Copy during cloning
  virtual Space*
  copy(void) {
    return new MultiKnapsack(*this);
  }

  virtual void constrain(const Space& _b) { // compute the bound at each leaf node giving a solution
    const MultiKnapsack& b = static_cast<const MultiKnapsack&>(_b);

    // We impose the constraint z >= current sol
    rel(*this, z >= b.z);

    if (activate_bound_computation){
      float copy_learning_rate = learning_rate;
      int nb_items = spec.nb_items();
      int nb_constraints = spec.nb_constraints();
      int weights[nb_items];
      int val[nb_items];

      for (int i = 0; i < nb_items; i++) {
        weights[i] = spec.weight(0, i, nb_items, nb_constraints);
        val[i] = spec.profit(i);
      }

      int rows = nb_items;
      int cols = nb_constraints;

      // Allocate memory for 2D array
      // value of the variables of the solution to update the multipliers
      int** value_var_solution = new int*[rows];
      for (int i = 0; i < rows; ++i) {
          value_var_solution[i] = new int[cols];
      }

      float final_bound = 0.0f;
      float bound_test[K];

      for (int k=0; k<K; k++) { // We repeat the dynamic programming algo to solve the knapsack problem 
                                // and at each iteration we update the value of the Lagrangian multipliers
        float bound_iter = 0.0f;
        std::vector<SubProblem> subproblems;
        // we create one subproblem for each knapsack constraint
        for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
          SubProblem subproblem;
          subproblem.weights_sub = new int[nb_items];
          subproblem.val_sub = new float[nb_items];
          subproblem.capacity = spec.capacity(idx_constraint, nb_items);
          for (int i = 0; i < nb_items; i++) {
            subproblem.weights_sub[i] = spec.weight(idx_constraint, i, nb_items, nb_constraints);
            if (idx_constraint == 0) {
              subproblem.val_sub[i] = spec.profit(i) + multipliers[i][idx_constraint];
            }
            else {
              subproblem.val_sub[i] = -multipliers[i][idx_constraint];
            }        
          }
          subproblem.idx_constraint = idx_constraint;
          subproblems.push_back(subproblem);
        }

        for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
          SubProblem subproblem = subproblems[id_subproblem];
          float bound = dp_knapsack(subproblem.capacity, 
                                    subproblem.weights_sub, 
                                    subproblem.val_sub, 
                                    nb_items, 
                                    nb_constraints, 
                                    subproblem.idx_constraint, 
                                    value_var_solution, 
                                    false);
          bound_iter += bound; // sum all the bound of the knapsack sub-problem to update the multipliers
        }
        final_bound = bound_iter;
        bound_test[k] = bound_iter;

        if (k >= 4) { // we divide by 2 the learning rate if the bound doesn't change 5 times in a row
          float b1 = bound_test[k];
          float b2 = bound_test[k-1];
          float b3 = bound_test[k-2];
          float b4 = bound_test[k-3];
          float b5 = bound_test[k-4];

          bool cond1 = (fabs(b1 - b2) <= 0.0001 * std::max(fabs(b1), fabs(b2)));
          bool cond2 = (fabs(b2 - b3) <= 0.0001 * std::max(fabs(b2), fabs(b3)));
          bool cond3 = (fabs(b3 - b4) <= 0.0001 * std::max(fabs(b3), fabs(b4)));
          bool cond4 = (fabs(b4 - b5) <= 0.0001 * std::max(fabs(b4), fabs(b5)));

          if (cond1 && cond2 && cond3 && cond4) {
            learning_rate = learning_rate/2;
          }
        }

        // TODO : update the multipliers (article method)

        // Update the multipliers (Quentin method)
        for (int i = 0; i < rows; ++i) {
          float sum = 0;
          for (int j = 1; j < cols; ++j) {
            multipliers[i][j] = multipliers[i][j] - learning_rate * (value_var_solution[i][0] - value_var_solution[i][j]);
            sum += multipliers[i][j];
          }
          multipliers[i][0] = sum;
        }
      }

      // We impose the constraint z <= final_bound
      rel(*this, z <= final_bound);

      learning_rate = copy_learning_rate;

      for (int i = 0; i < rows; ++i) {
          delete[] value_var_solution[i];
      }

      delete[] value_var_solution;
    }
}

float dp_knapsack(int capacity, 
                  int weights[], 
                  float val[], 
                  int nb_items, 
                  int nb_constraints, 
                  int idx_constraint, 
                  int** value_var_solution, 
                  bool verbose=false) { 
  // work only for 0-1 knapsack
  // Use dynamic programming to solve a 0-1 knapsack problem
  // (cf article 'A Dynamic Programming Approach for Consistency and Propagation for Knapsack Constraints' by MICHAEL A. TRICK)
  int i, w;

  // Define the dimensions of the graph, reverse graph and reduced graph
  int rows = capacity+1;
  int cols = nb_items+1;

  // Allocate memory for the graph, reverse graph and reduced graph
  int** graph = new int*[rows];
  for (int i = 0; i < rows; ++i) {
      graph[i] = new int[cols]();
  }

  int** reverse_graph = new int*[rows];
  for (int i = 0; i < rows; ++i) {
      reverse_graph[i] = new int[cols]();
  }

  int** reduced_graph = new int*[rows];
  for (int i = 0; i < rows; ++i) {
      reduced_graph[i] = new int[cols]();
  }

  // we update the graph 
  for (i = 0; i <= nb_items; i++) {
    for (w = 0; w <= capacity; w++) {
      if (i==0 && w==0) { 
        graph[w][i] = 1; // initialize the first row and column to 0
      } else if (i>0) {
          if (graph[w][i-1] == 1){
            if (w + weights[i-1]*0 <= capacity) { // test for xi = 0
              graph[w+weights[i-1]*0][i] = 1;
            } 
            if (w + weights[i-1]*1 <= capacity){ // test for xi = 1
              graph[w + weights[i-1]*1][i] = 1;
            }
        }
      }
    }
  }

  // We print the graph
  if (verbose){
    std::cout << "graph updated " << std::endl;
    std::cout << "graph:" << std::endl;
    for (int i = rows-1; i >= 0; i--) {
        for (int j = 0; j < cols; j++) {
            std::cout << graph[i][j] << " ";
        }
        std::cout << std::endl;
    }
  }

  // we check if there is at least one solution by testing if the last column is full of 0
  int test_last_column=0;
  for (w=0; w<=capacity; w++) {
    if (graph[w][nb_items] == 1) {
      break;
    }
    else {
      test_last_column++;
    }
  }
  if (test_last_column == capacity+1) {
    return 0;
  }

  // we update the reverse graph
  for (i = nb_items; i >= 0; i--) {
    for (w = 0; w <= capacity; w++) {
      if (i==nb_items) {
        reverse_graph[w][i] = 1; // initialize the last column to 1
      } else if (i<nb_items) {
        if (reverse_graph[w][i+1] == 1){
          if (w - weights[i]*0 >= 0) { // test for xi = 0
            reverse_graph[w-weights[i]*0][i] = 1;
          }
          if (w - weights[i]*1 >= 0){ // test for xi = 1
            reverse_graph[w-weights[i]*1][i] = 1;
          }
        }
      }
    }
  }

  // We print the reverse graph
  if (verbose){
    std::cout << "reverse graph updated " << std::endl;  
    std::cout << "reverse graph:" << std::endl;
    for (int i = rows-1; i >= 0; i--) {
        for (int j = 0; j < cols; j++) {
            std::cout << reverse_graph[i][j] << " ";
        }
        std::cout << std::endl;
    }
  }

  // get the reduced graph
  // get the id of the nodes present in the reduced graph
  std::vector<std::pair<int, int>> id_nodes_in_reduced_graph;
  for (i = 0; i <= nb_items; i++) {
    for (w = 0; w <= capacity; w++) {
      if (graph[w][i] == 1 && reverse_graph[w][i] == 1) {
        reduced_graph[w][i] = 1;
        id_nodes_in_reduced_graph.push_back(std::make_pair(w,i));
      }
      else {
        reduced_graph[w][i] = 0;
      }
    }
  }

  if (verbose){
    std::cout << "reduced graph updated " << std::endl;
    std::cout << "reduced graph:" << std::endl;
    for (int i = rows-1; i >= 0; i--) {
        for (int j = 0; j < cols; j++) {
            std::cout << reduced_graph[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "id_nodes_in_reduced_graph:" << std::endl;
    for (const auto& pair : id_nodes_in_reduced_graph) {
        std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
    }
  }

  std::vector<LabelEdge> label_edges;
  for (const auto& pair : id_nodes_in_reduced_graph) {
      // Search for the element in the vector
      int w = pair.first;
      int i = pair.second;
      // first we assign the 0 because if a item has a weight of 0 in this constraint we want to take it so we overwright with a 1
      if (weights[i-1] == 0) {
        label_edges.push_back({std::make_pair(w, i-1), pair, 1});
      }
      else {
        auto it2 = find(id_nodes_in_reduced_graph.begin(), id_nodes_in_reduced_graph.end(), std::make_pair(w, i-1));
        if (it2 != id_nodes_in_reduced_graph.end()) {
          label_edges.push_back({std::make_pair(w, i-1), pair, 0});
        }

        auto it = find(id_nodes_in_reduced_graph.begin(), id_nodes_in_reduced_graph.end(), std::make_pair(w-weights[i-1], i-1));
        if (it != id_nodes_in_reduced_graph.end()) {
          label_edges.push_back({std::make_pair(w-weights[i-1], i-1), pair, 1});
        }
      }
  }

  if (verbose){
    std::cout << "label_edges:" << std::endl;
    for (const auto& label_edge : label_edges) {
        std::cout << "(" << label_edge.pair1.first << ", " << label_edge.pair1.second << ") -> (" << label_edge.pair2.first << ", " << label_edge.pair2.second << ") : " << label_edge.label << std::endl;
    }
  }

  std::vector<CostEdge> cost_edges;
  for (const auto& label_edge : label_edges) {
    std::pair<int, int> pair1 = label_edge.pair1;
    std::pair<int, int> pair2 = label_edge.pair2;
    int label = label_edge.label;
    if (label == 1) {
      int idx_variable = pair2.second-1;
      cost_edges.push_back({pair1, pair2, val[idx_variable]});
    }
    else {
      cost_edges.push_back({pair1, pair2, 0});
    }
  }

  if (verbose){
    std::cout << "cost_edges:" << std::endl;
    for (const auto& cost_edge : cost_edges) {
        std::cout << "(" << cost_edge.pair1.first << ", " << cost_edge.pair1.second << ") -> (" << cost_edge.pair2.first << ", " << cost_edge.pair2.second << ") : " << cost_edge.cost << std::endl;
    }
  }

  std::vector<Path> paths;
  std::vector<float> costs;

  for (int i=cost_edges.size()-1; i>=0; i--) {
    std::pair<int, int> final_node = cost_edges[i].pair2;
    if (final_node.second == nb_items) { // it's an edge to the final node of a path
      std::vector<CostEdge> new_path;
      new_path.push_back(cost_edges[i]);
      float cost_path = cost_edges[i].cost;
      std::pair<int, int>node_to_find = cost_edges[i].pair1;
      for (int j=i-1; j>=0; j--) {
        if (cost_edges[j].pair2 == node_to_find) {
          new_path.push_back(cost_edges[j]);
          cost_path += cost_edges[j].cost;
          node_to_find = cost_edges[j].pair1;
        }
      }
      paths.push_back({new_path, cost_path});
      costs.push_back(cost_path);
    }
  }

  if (verbose){
    std::cout << "paths:" << std::endl;
    for (const auto& path : paths) {
        std::cout << "path:" << std::endl;
        for (const auto& cost_edge : path.path) {
            std::cout << "(" << cost_edge.pair1.first << ", " << cost_edge.pair1.second << ") -> (" << cost_edge.pair2.first << ", " << cost_edge.pair2.second << ") : " << cost_edge.cost << std::endl;
        }
        std::cout << "cost:" << path.cost << std::endl;
      }
  }

  // Get the best solution among all the paths in the reduced graph, the path with the higer cost
  Path best_path;
  auto max_cost_iterator = std::max_element(costs.begin(), costs.end());
  int index_max_cost = max_cost_iterator - costs.begin();
  best_path = paths[index_max_cost];

  
  if (verbose){
    std::cout << "best_path:" << best_path.cost << std::endl;
    for (const auto& cost_edge : best_path.path) {
        std::cout << "(" << cost_edge.pair1.first << ", " << cost_edge.pair1.second << ") -> (" << cost_edge.pair2.first << ", " << cost_edge.pair2.second << ") : " << cost_edge.cost << std::endl;
    }
  }

  for (const auto& cost_edge : best_path.path) {
    if (cost_edge.pair2.second>0) {
      if (idx_constraint == 0) { // in the first constraint we take the items for which the edge has a positive cost (c + sum(multipliers))
        if (cost_edge.cost > 0){
          value_var_solution[cost_edge.pair2.second-1][idx_constraint] = 1;   
        }
        else {
          value_var_solution[cost_edge.pair2.second-1][idx_constraint] = 0;
        }
      } 
      else { // in the others constraints we take the items for which the edge has a negative cost (-multiplier)
        if (cost_edge.cost < 0){
          value_var_solution[cost_edge.pair2.second-1][idx_constraint] = 1;
        }
        else {
          value_var_solution[cost_edge.pair2.second-1][idx_constraint] = 0;
        }
      }
    }
  }

  // get the bound of the knapsack sub-problem
  float bound = best_path.cost;

  // Deallocate memory
  for (int i = 0; i < rows; ++i) {
      delete[] graph[i];
      delete[] reverse_graph[i];
      delete[] reduced_graph[i];
  }
  delete[] graph;
  delete[] reverse_graph;
  delete[] reduced_graph;

  return bound;
}

  /// Print solution
  virtual void
  print(std::ostream& os) const {
    os << "z: " << z << std::endl;
    os << "x: " << x << std::endl;
  }
};

int main(int argc, char* argv[]) {
  bool activate_bound_computation = false;
  int K = 6;
  float learning_rate = 0.01f;
  float init_value_multipliers = 1.0f;
  OptionsKnapsack opt("MultiKnapsack", activate_bound_computation, K, learning_rate, init_value_multipliers);
  opt.instance(name[0]);
  opt.solutions(0);
  opt.parse(argc,argv);
  IntMaximizeScript::run<MultiKnapsack,BAB,OptionsKnapsack>(opt);
  return 0;
}

namespace {
  const int n1c1w1_a[] = {
    2, 28,
    1898, 440, 22507, 270, 14148, 3100, 4650, 30800, 615, 4975, 1160, 4225, 510, 11880, 479, 440, 490, 330, 110, 560, 24355, 2885, 11748, 4550, 750, 3720, 1950, 10500,
    600, 600,
    45, 0, 85, 150, 65, 95, 30, 0, 170, 0, 40, 25, 20, 0, 0, 25, 0, 0, 25, 0, 165, 0, 85, 0, 0, 0, 0, 100,
    30, 20, 125, 5, 80, 25, 35, 73, 12, 15, 15, 40, 5, 10, 10, 12, 10, 9, 0, 20, 60, 40, 50, 36, 49, 40, 19, 150,
    141278
  };

  // const int n1c1w1_a[] = {
  //   1, 4,
  //   1, 1, 1, 1,
  //   12,
  //   2, 3, 4, 5,
  //   141278
  // };

  // const std::vector<std::vector<int>>* mknps[] = {
  const int* mknps[] = {
    &n1c1w1_a[0],
  };

  const char* name[] = {
    "n1c1w1_a",
  };
}

