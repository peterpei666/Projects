#include "board.h"

class Game
{
private:
    Board board;
    bool gameOver;

public:
    Game(int rows,int cols,int mineCount);
    void run();
};
