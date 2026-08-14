import { useEffect, useState } from "react";

type Props = {
  puzzle: number[][];
  solution: number[][];
  resetKey: string;
};

export function SudokuBoard({ puzzle, solution, resetKey }: Props) {
  const [board, setBoard] = useState(() => puzzle.map((row) => [...row]));

  useEffect(() => setBoard(puzzle.map((row) => [...row])), [puzzle, resetKey]);

  const update = (row: number, column: number, raw: string) => {
    const value = /^[1-9]$/.test(raw) ? Number(raw) : 0;
    setBoard((current) => current.map((line, rowIndex) => line.map((cell, colIndex) => rowIndex === row && colIndex === column ? value : cell)));
  };
  const remaining = board
    .flatMap((row, rowIndex) => row.map((value, columnIndex): number => (
      puzzle[rowIndex][columnIndex] === 0 && value !== solution[rowIndex][columnIndex] ? 1 : 0
    )))
    .reduce((sum, value) => sum + value, 0);

  return (
    <div className="sudoku-panel">
      <div className="sudoku-status"><span>{remaining === 0 ? "Puzzle complete" : `${remaining} cells to resolve`}</span><button type="button" className="text-button" onClick={() => setBoard(puzzle.map((row) => [...row]))}>Reset board</button></div>
      <div className="sudoku-board" role="group" aria-label="Interactive Sudoku board">
        {board.flatMap((row, rowIndex) => row.map((value, columnIndex) => {
          const given = puzzle[rowIndex][columnIndex] !== 0;
          const state = given ? "given" : value === 0 ? "empty" : value === solution[rowIndex][columnIndex] ? "correct" : "conflict";
          return (
            <input
              key={`${rowIndex}-${columnIndex}`}
              aria-label={`Row ${rowIndex + 1} column ${columnIndex + 1}, ${given ? `given ${value}` : value ? `entered ${value}` : "empty"}`}
              className="sudoku-cell"
              data-row={rowIndex + 1}
              data-column={columnIndex + 1}
              data-state={state}
              disabled={given}
              inputMode="numeric"
              maxLength={1}
              value={value || ""}
              onChange={(event) => update(rowIndex, columnIndex, event.target.value)}
            />
          );
        }))}
      </div>
    </div>
  );
}
