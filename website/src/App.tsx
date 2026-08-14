import { currentRoute } from "./site";
import { AhdPage } from "./pages/AhdPage";
import { DocVqaPage } from "./pages/DocVqaPage";
import { HomePage } from "./pages/HomePage";
import { MathPage } from "./pages/MathPage";
import { PaperPage } from "./pages/PaperPage";
import { ScalingPage } from "./pages/ScalingPage";
import { SudokuPage } from "./pages/SudokuPage";
import { WebArenaPage } from "./pages/WebArenaPage";

export function App(){
  const route=currentRoute();
  if(route.includes("tasks/sudoku"))return <SudokuPage/>;
  if(route.includes("tasks/math"))return <MathPage/>;
  if(route.includes("tasks/docvqa"))return <DocVqaPage/>;
  if(route.includes("tasks/webarena"))return <WebArenaPage/>;
  if(route.includes("tasks/ahd"))return <AhdPage/>;
  if(route.includes("scaling"))return <ScalingPage/>;
  if(route.includes("paper"))return <PaperPage/>;
  return <HomePage/>;
}
