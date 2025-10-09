#include <iostream>
#include <unordered_map>
#include <string>
#include <time.h>
#include "relation.h"

void help()
{
    std::cout << "=== Relation Calculator Commands ===" << std::endl;
    std::cout << "create <name> <n> <pairs...>   : create new relation" << std::endl;
    std::cout << "delete <name>                  : delete existing relation" << std::endl;
    std::cout << "show <name>                    : show relation as set of pairs" << std::endl;
    std::cout << "domain <name>                  : show domain of relation" << std::endl;
    std::cout << "range <name>                   : show range of relation" << std::endl;
    std::cout << "reflexive <name>               : check if reflexive" << std::endl;
    std::cout << "symmetric <name>               : check if symmetric" << std::endl;
    std::cout << "antisymmetric <name>           : check if antisymmetric" << std::endl;
    std::cout << "transitive <name>              : check if transitive" << std::endl;
    std::cout << "equivalence <name>             : check if equivalence" << std::endl;
    std::cout << "partial_order <name>           : check if partial order" << std::endl;
    std::cout << "compose <r1> <r2> <newname>    : compute r1 after r2, save as new relation" << std::endl;
    std::cout << "power <name> <k> <newname>     : compute k-th power of relation" << std::endl;
    std::cout << "list                           : list all relations" << std::endl;
    std::cout << "exit                           : quit program" << std::endl;
}

int main()
{
    std::unordered_map<std::string, relation> relations;
    std::cout << "=== Relation Calculator ===" << std::endl;
    help();
    std::string cmd;
    while (true)
    {
        std::cout << "\n> ";
        if (!(std::cin >> cmd))
        {
            break;
        }
        if (cmd == "exit")
        {
            break;
        }
        else if (cmd == "help")
        {
            help();
        }
        else if (cmd == "create")
        {
            std::string name;
            int n, m;
            std::cin >> name >> n >> m;
            std::vector<std::bitset<N>> mat(n);
            for (int k = 0; k < m; k++)
            {
                int a, b;
                std::cin >> a >> b;
                if (a >= 0 && a < n && b >= 0 && b < n)
                {
                    mat[a].set(b);
                }
                else
                {
                    std::cerr << "Invalid pair (" << a << ", " << b << ")" << std::endl;
                }
            }
            relation r(n, mat.data());
            relations[name] = r;
            std::cout << "Relation '" << name << "' created." << std::endl;
        }
        else if (cmd == "delete")
        {
            std::string name;
            std::cin >> name;
            if (!relations.count(name))
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
            else
            {
                relations.erase(name);
                std::cout << "Relation '" << name << "' deleted." << std::endl;
            }
        }
        else if (cmd == "show")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << relations[name].to_string() << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "domain")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << relations[name].domain() << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "range")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << relations[name].range() << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "reflexive")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << (relations[name].is_reflexive() ? "Yes" : "No") << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "symmetric")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << (relations[name].is_symmetric() ? "Yes" : "No") << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "antisymmetric")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << (relations[name].is_antisymmetric() ? "Yes" : "No") << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "transitive")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << (relations[name].is_transitive() ? "Yes" : "No") << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "equivalence")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << (relations[name].is_equivalence() ? "Yes" : "No") << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "partial_order")
        {
            std::string name;
            std::cin >> name;
            if (relations.count(name))
            {
                std::cout << (relations[name].is_partial_order() ? "Yes" : "No") << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "compose")
        {
            std::string r1, r2, newname;
            std::cin >> r1 >> r2 >> newname;
            if (relations.count(r1) && relations.count(r2))
            {
                relations[newname] = relations[r1].after(relations[r2]);
                std::cout << "Relation '" << newname << "' = " << r1 << " after " << r2 << " created." << std::endl;
            }
            else
            {
                std::cout << "Missing relation." << std::endl;
            }
        }
        else if (cmd == "power")
        {
            std::string name, newname; unsigned long long k;
            std::cin >> name >> k >> newname;
            if (relations.count(name))
            {
                relations[newname] = relations[name].after(k);
                std::cout << "Relation '" << newname << "' = " << name << "^" << k << " created." << std::endl;
            }
            else
            {
                std::cout << "No relation named '" << name << "'." << std::endl;
            }
        }
        else if (cmd == "list")
        {
            std::cout << "Relations stored: ";
            for (auto &p : relations)
            {
                std::cout << p.first << " ";
            }
            std::cout << std::endl;
        }
        else
        {
            std::cout << "Unknown command. Type 'help' for usage." << std::endl;
        }
    }
    std::cout << "Bye!" << std::endl;
    return 0;
}
