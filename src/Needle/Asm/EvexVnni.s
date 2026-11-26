.intel_syntax noprefix
.text
# One entry point per weight width. Argument: pointer to a block of 8-byte fields
#   0 packed   8 rowBytes  16 norms  24 groups  32 stride  40 weightedScales  48 offsets
#  56 values  64 scaled   72 tables  80 y       88 blocks  96 prefetchDistance
# The first instruction (mov rax, rdi | mov rax, rcx) is patched per ABI at load time.
# Only ymm0-5 and ymm16-31 are written, so the Win64 callee-saved xmm6-15 stay intact.

.macro CORR acc, k
    vxorps \acc, \acc, \acc
    mov rdi, r10
.rept \k
    add rdi, r11
.endr
    mov rsi, rbx
.rept \k
    add rsi, r15
.endr
    mov rdx, [rax+40]
    mov r13, [rax+48]
    xor ecx, ecx
1:
    vpmovzxwd ymm4, [rdi + rcx*2]
    vpslld ymm4, ymm4, 13
    vmulps ymm4, ymm4, [rdx + rcx*4]
    vmovups [rsi + rcx*4], ymm4
    vfnmadd231ps \acc, ymm4, [r13 + rcx*4]
    add rcx, 8
    cmp rcx, [rax+32]
    jb 1b
.endm

.macro DOT acc
    vpxord ymm26, ymm26, ymm26
    vpxord ymm27, ymm27, ymm27
    vpdpbusd ymm26, ymm22, ymm18
    vpdpbusd ymm27, ymm23, ymm19
    vpdpbusd ymm26, ymm24, ymm20
    vpdpbusd ymm27, ymm25, ymm21
    vpaddd ymm26, ymm26, ymm27
    vcvtdq2ps ymm26, ymm26
    vbroadcastss ymm28, [r13 + rdx*4]
    vfmadd231ps \acc, ymm26, ymm28
.endm

.macro EXPAND bits, addr, addrhi
.if \bits == 2
    vmovdqu64 ymm22, \addr
    vpsrlw ymm24, ymm22, 4
    vpermb ymm23, ymm22, ymm17
    vpermb ymm22, ymm22, ymm16
    vpermb ymm25, ymm24, ymm17
    vpermb ymm24, ymm24, ymm16
.else
    vmovdqu64 ymm22, \addr
    vmovdqu64 ymm24, \addrhi
    vpsrlw ymm23, ymm22, 4
    vpsrlw ymm25, ymm24, 4
    vpermb ymm22, ymm22, ymm16
    vpermb ymm23, ymm23, ymm16
    vpermb ymm24, ymm24, ymm16
    vpermb ymm25, ymm25, ymm16
.endif
.endm

.macro KERNEL bits, gbytes
    mov rax, rdi
    push rbx
    push rbp
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    mov r8,  [rax]
    mov r9,  [rax+8]
    lea r12, [r9+r9*2]
    mov r10, [rax+16]
    mov r11, [rax+24]
    add r11, r11
    mov r15, [rax+32]
    shl r15, 2
    mov rbx, [rax+64]
    mov rbp, [rax+96]
    mov rcx, [rax+72]
    vmovdqu64 ymm16, [rcx]
    vmovdqu64 ymm17, [rcx+32]
    cmp qword ptr [rax+88], 0
    je 9f
2:
    CORR ymm0, 0
    CORR ymm1, 1
    CORR ymm2, 2
    CORR ymm3, 3
    xor edx, edx
    xor ecx, ecx
    mov rsi, [rax+56]
3:
    test ecx, 63
    jnz 4f
    lea rdi, [r8 + rcx]
    add rdi, rbp
    prefetcht0 [rdi]
    prefetcht0 [rdi + r9]
    prefetcht0 [rdi + r9*2]
    prefetcht0 [rdi + r12]
4:
    vmovdqu64 ymm18, [rsi]
    vmovdqu64 ymm19, [rsi+32]
    vmovdqu64 ymm20, [rsi+64]
    vmovdqu64 ymm21, [rsi+96]
    lea rdi, [r8 + rcx]
    mov r13, rbx
    EXPAND \bits, "[rdi]", "[rdi+32]"
    DOT ymm0
    add r13, r15
    EXPAND \bits, "[rdi+r9]", "[rdi+r9+32]"
    DOT ymm1
    add r13, r15
    EXPAND \bits, "[rdi+r9*2]", "[rdi+r9*2+32]"
    DOT ymm2
    add r13, r15
    EXPAND \bits, "[rdi+r12]", "[rdi+r12+32]"
    DOT ymm3
    add rcx, \gbytes
    add rsi, 128
    inc rdx
    cmp rdx, [rax+24]
    jb 3b
    vhaddps ymm4, ymm0, ymm1
    vhaddps ymm5, ymm2, ymm3
    vhaddps ymm4, ymm4, ymm5
    vextractf128 xmm5, ymm4, 1
    vaddps xmm4, xmm4, xmm5
    mov rcx, [rax+80]
    vmovups [rcx], xmm4
    add rcx, 16
    mov [rax+80], rcx
    lea r8, [r8 + r9*4]
    lea r10, [r10 + r11*4]
    dec qword ptr [rax+88]
    jnz 2b
9:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbp
    pop rbx
    ret
.endm

.globl block2
block2:
    KERNEL 2, 32
.balign 64, 0xcc
.globl block4
block4:
    KERNEL 4, 64
.balign 64, 0xcc
