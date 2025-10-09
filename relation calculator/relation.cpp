#include "relation.h"

std::bitset<N> relation::get_domain() const
{
    std::bitset<N> ans;
    for (int i = 0; i < n; i++)
    {
        if (mp[i].any())
        {
            ans.set(i);
        }
    }
    return ans;
}
    
std::bitset<N> relation::get_range() const
{
    std::bitset<N> ans;
    for (int i = 0; i < n; i++)
    {
        ans |= mp[i];
    }
    return ans;
}

relation::relation() {}

relation::relation(int n, std::bitset<N> m[])
{
    this->n = n;
    for (int i = 0; i < n; i++)
    {
        mp[i] = m[i];
    }
}
    
std::string relation::domain() const
{
    std::string s = "{";
    std::bitset<N> ans = get_domain();
    bool first = true;
    for (int i = 0; i < n; i++)
    {
        if (ans.test(i))
        {
            if (first)
            {
                first = false;
            }
            else
            {
                s += ", ";
            }
            s += std::to_string(i);
        }
    }
    s += "}";
    return s;
}
    
std::string relation::range() const
{
    std::string s = "{";
    std::bitset<N> ans = get_range();
    bool first = true;
    for (int i = 0; i < n; i++)
    {
        if (ans.test(i))
        {
            if (first)
            {
                first = false;
            }
            else
            {
                s += ", ";
            }
            s += std::to_string(i);
        }
    }
    s += "}";
    return s;
}
    
std::string relation::to_string() const
{
    std::string s = "{";
    bool first = true;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (mp[i].test(j))
            {
                if (first)
                {
                    first = false;
                }
                else
                {
                    s += ", ";
                }
                s += "(" + std::to_string(i) + ", "+ std::to_string(j) + ")";
            }
        }
    }
    s += "}";
    return s;
}
    
bool relation::is_reflexive() const
{
    for (int i = 0; i < n; i++)
    {
        if (mp[i].test(i) == false)
        {
            return false;
        }
    }
    return true;
}
    
bool relation::is_symmetric() const
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (mp[i].test(j) && !mp[j].test(i))
            {
                return false;
            }
        }
    }
    return true;
}

bool relation::is_antisymmetric() const
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (mp[i].test(j) && mp[j].test(i))
            {
                return false;
            }
        }
    }
    return true;
}

bool relation::is_transitive() const
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (mp[i].test(j))
            {
                if ((mp[i] | mp[j]) != mp[i])
                {
                    return false;
                }
            }
        }
    }
    return true;
}
    
bool relation::is_equivalence() const
{
    if (is_reflexive() && is_symmetric() && is_transitive())
    {
        return true;
    }
    return false;
}
    

bool relation::is_partial_order() const
{
    if (is_reflexive() && is_antisymmetric() && is_transitive())
    {
        return true;
    }
    return false;
}

relation relation::after(const relation& other) const
{
    std::bitset<N> temp[N];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (mp[i].test(j))
            {
                temp[i] |= other.mp[j];
            }
        }
    }
    return relation(n, temp);
}

relation relation::after(unsigned long long k) const
{
    if (k == 1)
    {
        return *this;
    }
    k--;
    relation temp = *this;
    relation ans = *this;
    while (k)
    {
        if (k & 1)
        {
            ans = ans.after(temp);
        }
        k >>= 1;
        temp = temp.after(temp);
    }
    return ans;
}
