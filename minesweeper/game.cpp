#include "game.h"

Game::Game(int rows, int cols, int mineCount): board(rows, cols, mineCount)
{
    gameOver=false;
}

void Game::run()
{
    while(!gameOver)
    {
        board.display();
        int option;
        std::cout<<"1 for open, 2 for flag"<<std::endl;
        std::cin>>option;
        int r,c;
        std::cin>>r>>c;
        switch (option)
        {
            case 1:
                board.openCell(r,c);
                break;
            case 2:
                board.toggleFlag(r,c);
                break;
            default:
                std::cout<<"Invalid command"<<std::endl;
                break;
        }
        if(board.lose)
        {
            gameOver=true;
        }
        if(board.isWin())
        {
            std::cout<<"You win"<<std::endl;
            gameOver=true;
        }
    }
    board.displayall();
}

