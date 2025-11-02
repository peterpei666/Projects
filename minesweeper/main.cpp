#include "game.h"

int main(void)
{
    int r,c,t;
    std::cout<<"input the number of rows, columns and mines"<<std::endl;
    std::cin>>r>>c>>t;
    while (r<3||r>50||c<3||c>50||t<=0||c>=r*c)
    {
        std::cout<<"Invalid number"<<std::endl;
        std::cout<<"input the number of rows, columns and mines"<<std::endl;
        std::cin>>r>>c>>t;
    }
    Game game(r,c,t);
    game.run();
    return 0;
}
