#include <iostream>
#include <bitset>
#include <random>
#include <stdexcept>

class Board
{
private:
    int rows,cols,totalMines;
    std::bitset<52> mines[52];
    std::bitset<50> revealed[50];
    std::bitset<50> flagged[50];
    int neighborCount[50][50];
    bool valid(int row, int col) const;

public:
    Board(int rows, int cols, int mineCount);
    bool lose;
    void display();
    void displayall();
    void openCell(int row, int col);
    void toggleFlag(int row, int col);
    bool isWin() const;
};


