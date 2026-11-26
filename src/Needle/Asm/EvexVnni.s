.intel_syntax noprefix
.text
# One entry point per weight width. Argument: pointer to a block of 8-byte fields
#   0 packed   8 rowBytes  16 norms  24 groups  32 stride  40 weightedScales  48 offsets
#  56 values  64 scaled   72 tables  80 y       88 count   96 prefetchDistance
# 104 cols   112 ldy (both read by the pair kernels only)
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

# Two rows x four tokens (Gemv.Pair4), for prefill tiles. Extra fields: 104 cols, 112 ldy (floats).
# scaled holds eight rows of stride floats: (row r, token t) at (4r + t) * stride.
# GPRs: r8 packed pair base, r9 rowBytes, r10 norms pair base, r11 norm row bytes, rbx scaled,
# rbp prefetch distance, r12 cols (3 * stride bytes during corrections), r13 3 * cols,
# r15 stride bytes, rcx/rdx/rsi/rdi/r14 scratch.

.macro PCORR
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vpxord ymm28, ymm28, ymm28
    vpxord ymm29, ymm29, ymm29
    vpxord ymm30, ymm30, ymm30
    vpxord ymm31, ymm31, ymm31
    lea r12, [r15 + r15*2]
    mov r13, r10
    mov rsi, [rax+40]
    mov rdi, [rax+48]
    mov r14, rbx
    lea rdx, [rbx + r15*4]
    xor ecx, ecx
1:
    vpmovzxwd ymm18, [r13]
    vpslld ymm18, ymm18, 13
    vpmovzxwd ymm19, [r13 + r11]
    vpslld ymm19, ymm19, 13
    PCORR1 0, ymm0, ymm28, "[rsi]", "[rdi]", "[r14]", "[rdx]"
    PCORR1 1, ymm1, ymm29, "[rsi + r15]", "[rdi + r15]", "[r14 + r15]", "[rdx + r15]"
    PCORR1 2, ymm2, ymm30, "[rsi + r15*2]", "[rdi + r15*2]", "[r14 + r15*2]", "[rdx + r15*2]"
    PCORR1 3, ymm3, ymm31, "[rsi + r12]", "[rdi + r12]", "[r14 + r12]", "[rdx + r12]"
    add r13, 16
    add rsi, 32
    add rdi, 32
    add r14, 32
    add rdx, 32
    add rcx, 8
    cmp rcx, [rax+32]
    jb 1b
.endm

.macro PCORR1 t, acc0, acc1, ws, off, s0, s1
    vmulps ymm20, ymm18, \ws
    vmovups \s0, ymm20
    vfnmadd231ps \acc0, ymm20, \off
    vmulps ymm21, ymm19, \ws
    vmovups \s1, ymm21
    vfnmadd231ps \acc1, ymm21, \off
.endm

# PDOT acc, x, xhi-offsets via memory operands, scale address
.macro PDOT acc, x0, x1, x2, x3, scale
    vpxord ymm22, ymm22, ymm22
    vpxord ymm23, ymm23, ymm23
    vpdpbusd ymm22, ymm18, \x0
    vpdpbusd ymm23, ymm19, \x1
    vpdpbusd ymm22, ymm20, \x2
    vpdpbusd ymm23, ymm21, \x3
    vpaddd ymm22, ymm22, ymm23
    vcvtdq2ps ymm22, ymm22
    vbroadcastss ymm24, \scale
    vfmadd231ps \acc, ymm22, ymm24
.endm

.macro PEXPAND bits, addr, addrhi
.if \bits == 2
    vmovdqu64 ymm18, \addr
    vpsrlw ymm20, ymm18, 4
    vpermb ymm19, ymm18, ymm17
    vpermb ymm18, ymm18, ymm16
    vpermb ymm21, ymm20, ymm17
    vpermb ymm20, ymm20, ymm16
.else
    vmovdqu64 ymm18, \addr
    vmovdqu64 ymm20, \addrhi
    vpsrlw ymm19, ymm18, 4
    vpsrlw ymm21, ymm20, 4
    vpermb ymm18, ymm18, ymm16
    vpermb ymm19, ymm19, ymm16
    vpermb ymm20, ymm20, ymm16
    vpermb ymm21, ymm21, ymm16
.endif
.endm

.macro PTOKENS a0, a1, a2, a3
    PDOT \a0, "[rsi]", "[rsi+32]", "[rsi+64]", "[rsi+96]", "[r14]"
    PDOT \a1, "[rsi+r12]", "[rsi+r12+32]", "[rsi+r12+64]", "[rsi+r12+96]", "[r14+r15]"
    PDOT \a2, "[rsi+r12*2]", "[rsi+r12*2+32]", "[rsi+r12*2+64]", "[rsi+r12*2+96]", "[r14+r15*2]"
    PDOT \a3, "[rsi+r13]", "[rsi+r13+32]", "[rsi+r13+64]", "[rsi+r13+96]", "[rdi+r15]"
.endm

# Store lanes of xmm4 to y[0], y[ldy], y[2 ldy], y[3 ldy] with y in rcx and ldy bytes in rdx.
.macro PSTORE
    vmovss [rcx], xmm4
    vextractps [rcx + rdx], xmm4, 1
    vextractps [rcx + rdx*2], xmm4, 2
    lea rsi, [rcx + rdx*2]
    vextractps [rsi + rdx], xmm4, 3
.endm

.macro PAIR4 bits, gbytes
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
    PCORR
    mov r12, [rax+104]
    lea r13, [r12 + r12*2]
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
4:
    lea r14, [rbx + rdx*4]
    lea rdi, [r14 + r15*2]
    PEXPAND \bits, "[r8 + rcx]", "[r8 + rcx + 32]"
    PTOKENS ymm0, ymm1, ymm2, ymm3
    lea r14, [r14 + r15*4]
    lea rdi, [r8 + rcx]
    PEXPAND \bits, "[rdi + r9]", "[rdi + r9 + 32]"
    lea rdi, [r14 + r15*2]
    PTOKENS ymm28, ymm29, ymm30, ymm31
    add rcx, \gbytes
    add rsi, 128
    inc rdx
    cmp rdx, [rax+24]
    jb 3b
    mov rcx, [rax+80]
    mov rdx, [rax+112]
    shl rdx, 2
    vhaddps ymm4, ymm0, ymm1
    vhaddps ymm5, ymm2, ymm3
    vhaddps ymm4, ymm4, ymm5
    vextractf128 xmm5, ymm4, 1
    vaddps xmm4, xmm4, xmm5
    PSTORE
    vmovaps ymm0, ymm28
    vmovaps ymm1, ymm29
    vmovaps ymm2, ymm30
    vmovaps ymm3, ymm31
    vhaddps ymm4, ymm0, ymm1
    vhaddps ymm5, ymm2, ymm3
    vhaddps ymm4, ymm4, ymm5
    vextractf128 xmm5, ymm4, 1
    vaddps xmm4, xmm4, xmm5
    add rcx, 4
    PSTORE
    add rcx, 4
    mov [rax+80], rcx
    lea r8, [r8 + r9*2]
    lea r10, [r10 + r11*2]
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

.globl pair2
pair2:
    PAIR4 2, 32
.balign 64, 0xcc
.globl pair4
pair4:
    PAIR4 4, 64
.balign 64, 0xcc
