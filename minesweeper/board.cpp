#include "board.h"

int randomInt(int l, int r)
{
    if (l > r) throw std::invalid_argument("Lower bound cannot be greater than upper bound");
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(l, r);
    return dist(gen);
}

Board::Board(int r, int c, int mineCount)
{
    totalMines=mineCount;
    rows=r;
    cols=c;
    int cnt=0;
    while(cnt<mineCount)
    {
        int x=randomInt(1,rows);
        int y=randomInt(1,cols);
        if(mines[x].test(y)==false)
        {
            mines[x].set(y);
            cnt++;
        }
    }
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            if(mines[i+1].test(j+1))
            {
                neighborCount[i][j]=-1;
            }
            else
            {
                int cnt=0;
                for(int m=i;m<=i+2;m++)
                {
                    for(int n=j;n<=j+2;n++)
                    {
                        if(Board::mines[m].test(n))
                        {
                            cnt++;
                        }
                    }
                }
                neighborCount[i][j]=cnt;
            }
        }
    }
    lose=false;
    for(int i=0;i<50;i++)
    {
        revealed[i].reset();
        flagged[i].reset();
    }
}

void Board::display()
{
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            if(flagged[i].test(j)&&revealed[i].test(j)==false)
            {
                std::cout<<"F";
            }
            else if(revealed[i].test(j)==false)
            {
                std::cout<<"*";
            }
            else if(neighborCount[i][j]==0)
            {
                std::cout<<" ";
            }
            else
            {
                std::cout<<neighborCount[i][j];
            }
        }
        std::cout<<std::endl;
    }
}

void Board::displayall()
{
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            if(neighborCount[i][j]==-1)
            {
                std::cout<<"*";
            }
            else
            {
                std::cout<<neighborCount[i][j];
            }
        }
        std::cout<<std::endl;
    }
}

bool Board::valid(int row,int col) const
{
    if(row<0||row>=rows||col<0||col>=cols)
    {
        return false;
    }
    return true;
}

void Board::toggleFlag(int row, int col)
{
    if(valid(row,col))
    {
        if(revealed[row].test(col))
        {
            return;
        }
        flagged[row].flip(col);
    }
    else
    {
        std::cout<<"Invalid position"<<std::endl;
    }
}

void Board::openCell(int row, int col)
{
    if(valid(row,col)==false)
    {
        std::cout<<"Invalid position"<<std::endl;
        return;
    }
    if(revealed[row].test(col))
    {
        std::cout<<"The position has been opened"<<std::endl;
        return;
    }
    if(neighborCount[row][col]==-1)
    {
        lose=true;
        std::cout<<"You lose"<<std::endl;
        return;
    }
    if(neighborCount[row][col])
    {
        revealed[row].set(col);
        return;
    }
    int dir[4][2]={-1,0,0,-1,0,1,1,0};
    revealed[row].set(col);
    std::queue<std::pair<int,int>> q;
    q.push({row,col});
    while(!q.empty())
    {
        auto [x,y]=q.front();
        q.pop();
        for(int d=0;d<4;d++)
        {
            int nx=x+dir[d][0];
            int ny=y+dir[d][1];
            if(valid(nx,ny)&&revealed[nx].test(ny)==false)
            {
                revealed[nx].set(ny);
                if(neighborCount[nx][ny]==0)
                {
                    q.push({nx,ny});
                }
            }
        }
    }
}

bool Board::isWin() const
{
    int cnt=0;
    for(int i=0;i<rows;i++)
    {
        cnt+=(int)revealed[i].count();
    }
    return cnt+totalMines==rows*cols;
}
