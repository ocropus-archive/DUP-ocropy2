set nocompatible
filetype off

highlight ColorColumn guibg=#ccffff ctermbg=gray
set colorcolumn=80
let c_space_errors=1
let mapleader=","
let maplocalleader="\\"
set nows
set autoindent
set showmatch
set ignorecase
set ttyfast
set guifont="Courier 10"
set softtabstop=4
set shiftwidth=4
set tabstop=8
set expandtab
set enc=utf-8
set fenc=utf-8
set termencoding=utf-8
set smartindent
set t_Co=256
syn on
set number
set switchbuf=useopen
" set equalprg=clang-format

nmap <M-=> :w<CR>:make<CR>:copen<CR><C-W><C-W>
nmap <M-/> :cn<CR>
command W w
command Fc :FormatCode clang-format
command Fp :FormatCode pyformat
map <localleader>b :FormatCode<cr>
map <C-N> :n<CR>
imap <C-A> <ESC>0i
nmap <C-A> <ESC>0
cmap <C-A> <Home> 
nmap <C-E> $
imap <C-E> <ESC>A
nnoremap ZZ :w<cr>:qa<cr>
nnoremap Zz :qa<cr>
