#include <bitset>

#define N 500

class relation
{
private:
    int n;
    std::bitset<N> mp[N];
    std::bitset<N> get_domain() const;
    std::bitset<N> get_range() const;
    
public:
    relation();
    relation(int n, std::bitset<N> m[]);
    std::string domain() const;
    std::string range() const;
    std::string to_string() const;
    bool is_reflexive() const;
    bool is_symmetric() const;
    bool is_antisymmetric() const;
    bool is_transitive() const;
    bool is_equivalence() const;
    bool is_partial_order() const;
    relation after(const relation& other) const;
    relation after(unsigned long long k) const;
};

