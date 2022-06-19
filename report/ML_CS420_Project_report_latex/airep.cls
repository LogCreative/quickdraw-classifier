\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{airep}[]
\LoadClass{ctexart}

% 不超过7页，简单做了

% Equations

\RequirePackage[]{amsmath, amssymb}

% change the layout

\usepackage{geometry}
\geometry{left=2cm,right=2cm,top=3cm,bottom=3cm}

% redefine the numbering
\ctexset{
    section = {
        % number = \chinese{section},
        format = \bfseries\large\raggedright,
    }
}

% add color to titles
\usepackage{xcolor}
\definecolor{maintheme}{RGB}{70,130,180}
\ctexset {
    part ={
        format += \color{maintheme!50!black}
    },
    section = {
        format += \color{maintheme!70!black}\raggedright
    },
    subsection = {
        format += \color{maintheme!90!black}\raggedright
    },
    subsubsection = {
        format += \color{maintheme}
    }
}
\renewcommand\sectionmark[1]{%
\markright{\color{maintheme}\CTEXifname{\CTEXthesection~~~~}{}#1}}

% Reference

\RequirePackage[backend=biber, style=gb7714-2015]{biblatex}

% Figures

\RequirePackage[]{graphicx}
\graphicspath{{img/}}
\RequirePackage[]{float} % Provide H environment

% listings

\RequirePackage[]{listings}
\lstset{
    language=python,
    basicstyle=\ttfamily,
    columns=flexible,
}

% Tables
\RequirePackage[]{booktabs}

\RequirePackage[]{tikz}
\usetikzlibrary{quotes,arrows.meta,shapes,calc}
\usepackage{tikz-3dplot}
\usetikzlibrary{positioning,mindmap}
\usepackage{bm}
\usepackage{pgfplots}
\usepackage{caption}

\usepackage{subfigure}

% Hyperref

\RequirePackage[colorlinks,citecolor=maintheme]{hyperref}