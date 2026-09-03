// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;

/*
    Copyright 2021 0KIMS association.

    * `solidity-verifiers` added comment
        This file is a template built out of [snarkJS](https://github.com/iden3/snarkjs) groth16 verifier.
        See the original ejs template [here](https://github.com/iden3/snarkjs/blob/master/templates/verifier_groth16.sol.ejs)
    *

    snarkJS is a free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    snarkJS is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
    License for more details.

    You should have received a copy of the GNU General Public License
    along with snarkJS. If not, see <https://www.gnu.org/licenses/>.
*/

contract Groth16Verifier {
    // Scalar field size
    uint256 constant r    = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    // Base field size
    uint256 constant q   = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    // Verification Key data
    uint256 constant alphax  = 17682257115974194825317626468286310123646643203864866303159635189181674825277;
    uint256 constant alphay  = 6198153535947025019038829574974605416029115198273557013189882401038268644144;
    uint256 constant betax1  = 1994503134453860903905698780652532412870315380568170510275574608526935978550;
    uint256 constant betax2  = 4983279845017530179485275986029636658699566993750798425913401727966042146348;
    uint256 constant betay1  = 13612389724867715052481110210571479168482705012779311190006966934037602216758;
    uint256 constant betay2  = 13196215940676755829172966353302050895658820810007955654756788509349816624383;
    uint256 constant gammax1 = 14150804104482061557921612106968696070891464569059046466104131057727920836520;
    uint256 constant gammax2 = 15983410902366336486422716651813089461556848496653534300324107157395999837160;
    uint256 constant gammay1 = 2915402442420468586820914369209107149607161723597578893682332263613450987476;
    uint256 constant gammay2 = 13952209508665592965424177378403629534387899544485116753564681848922274410487;
    uint256 constant deltax1 = 10996140775981799605431025341251907019968339896649002643171922976587931669304;
    uint256 constant deltax2 = 21058172394203306541901416712978389508483434112610500110247350743886666344737;
    uint256 constant deltay1 = 13518818754852975318198463574747909033383597688079644014660115824428916219203;
    uint256 constant deltay2 = 8145081633926448896953464118630890737681560196603140562075851350295402615424;

    
    uint256 constant IC0x = 5686670915395052279270746352674237275547752874188714244336600580516043434837;
    uint256 constant IC0y = 21291411047143518659370260033674624562488116866885786322045084042146183148330;
    
    uint256 constant IC1x = 11428866579677418843260337582789002350389473540981860977764508116340164885542;
    uint256 constant IC1y = 1117349298759137632343126185227729179615557312415160167774435397753330070626;
    
    uint256 constant IC2x = 21607670794206332082234907981201780240140292898732921905224977581694150519819;
    uint256 constant IC2y = 15963163845166768075438330518096426405337292860210527690764673373943071125438;
    
    uint256 constant IC3x = 21034531753908081910416078950243630938875367273518890521411782277060560801266;
    uint256 constant IC3y = 2749724267491163929651859044515584693148336096161742116838610454272091623660;
    
    uint256 constant IC4x = 20086893035786143028131952347596927206053180361036679705408188117255751518106;
    uint256 constant IC4y = 14472000249332642196417630446546203946802539959520342311525166525387072454739;
    
    uint256 constant IC5x = 20023260170117165950830585193141103701602544987587155121846845322383892424526;
    uint256 constant IC5y = 11413830676694618639649256753979396959719767225482222102830217712587341648961;
    
    uint256 constant IC6x = 12821629357899450045015952767797224938620043596194715409851056198688779395425;
    uint256 constant IC6y = 19199097872929231093903754209976123848731226525858217968160523271840767078538;
    
    uint256 constant IC7x = 18462825782915477809650408875798468670807419764335960309160875492264115912221;
    uint256 constant IC7y = 20468899682761982394060479935487796161046380541122260398249349436426202059114;
    
    uint256 constant IC8x = 5554084456604088281645232517377198195569563697256330282768586902901708071137;
    uint256 constant IC8y = 11615404582745096033530641648953043667889374399795693688144739673645467104967;
    
    uint256 constant IC9x = 19943425459175885018522622568732226608817546348578677987681684269347044976272;
    uint256 constant IC9y = 9494078096402639197372053804976884247282677431824899711763679577516264220962;
    
    uint256 constant IC10x = 20822841006934491539966174765497970275776559200375935867436931684723234335105;
    uint256 constant IC10y = 16498911885455808913937172185891469657860297459912676608039033107082200606304;
    
    uint256 constant IC11x = 2175160426262611853605884576505555314135490565984270773603655742534595334323;
    uint256 constant IC11y = 1376775228297989047236961672175770282643341128072171524068360549285142264090;
    
    uint256 constant IC12x = 2138177516570641332159030437680905359492443822443349270808330276114598291358;
    uint256 constant IC12y = 6154204678989242800551737609542805907600492807447418705460246322141836726856;
    
    uint256 constant IC13x = 19873274587659078600092486036344070810712724230994211420498346568804745736750;
    uint256 constant IC13y = 8245332741545522331323715811234063158503918455002171119972092868738205689676;
    
    
    // Memory data
    uint16 constant pVk = 0;
    uint16 constant pPairing = 128;

    uint16 constant pLastMem = 896;

    function verifyProof(uint[2] calldata _pA, uint[2][2] calldata _pB, uint[2] calldata _pC, uint[13] calldata _pubSignals) public view returns (bool) {
        assembly {
            function checkField(v) {
                if iszero(lt(v, r)) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }
            
            // G1 function to multiply a G1 value(x,y) to value in an address
            function g1_mulAccC(pR, x, y, s) {
                let success
                let mIn := mload(0x40)
                mstore(mIn, x)
                mstore(add(mIn, 32), y)
                mstore(add(mIn, 64), s)

                success := staticcall(sub(gas(), 2000), 7, mIn, 96, mIn, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }

                mstore(add(mIn, 64), mload(pR))
                mstore(add(mIn, 96), mload(add(pR, 32)))

                success := staticcall(sub(gas(), 2000), 6, mIn, 128, pR, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }

            function checkPairing(pA, pB, pC, pubSignals, pMem) -> isOk {
                let _pPairing := add(pMem, pPairing)
                let _pVk := add(pMem, pVk)

                mstore(_pVk, IC0x)
                mstore(add(_pVk, 32), IC0y)

                // Compute the linear combination vk_x
                
                
                g1_mulAccC(_pVk, IC1x, IC1y, calldataload(add(pubSignals, 0)))
                g1_mulAccC(_pVk, IC2x, IC2y, calldataload(add(pubSignals, 32)))
                g1_mulAccC(_pVk, IC3x, IC3y, calldataload(add(pubSignals, 64)))
                g1_mulAccC(_pVk, IC4x, IC4y, calldataload(add(pubSignals, 96)))
                g1_mulAccC(_pVk, IC5x, IC5y, calldataload(add(pubSignals, 128)))
                g1_mulAccC(_pVk, IC6x, IC6y, calldataload(add(pubSignals, 160)))
                g1_mulAccC(_pVk, IC7x, IC7y, calldataload(add(pubSignals, 192)))
                g1_mulAccC(_pVk, IC8x, IC8y, calldataload(add(pubSignals, 224)))
                g1_mulAccC(_pVk, IC9x, IC9y, calldataload(add(pubSignals, 256)))
                g1_mulAccC(_pVk, IC10x, IC10y, calldataload(add(pubSignals, 288)))
                g1_mulAccC(_pVk, IC11x, IC11y, calldataload(add(pubSignals, 320)))
                g1_mulAccC(_pVk, IC12x, IC12y, calldataload(add(pubSignals, 352)))
                g1_mulAccC(_pVk, IC13x, IC13y, calldataload(add(pubSignals, 384)))

                // -A
                mstore(_pPairing, calldataload(pA))
                mstore(add(_pPairing, 32), mod(sub(q, calldataload(add(pA, 32))), q))

                // B
                mstore(add(_pPairing, 64), calldataload(pB))
                mstore(add(_pPairing, 96), calldataload(add(pB, 32)))
                mstore(add(_pPairing, 128), calldataload(add(pB, 64)))
                mstore(add(_pPairing, 160), calldataload(add(pB, 96)))

                // alpha1
                mstore(add(_pPairing, 192), alphax)
                mstore(add(_pPairing, 224), alphay)

                // beta2
                mstore(add(_pPairing, 256), betax1)
                mstore(add(_pPairing, 288), betax2)
                mstore(add(_pPairing, 320), betay1)
                mstore(add(_pPairing, 352), betay2)

                // vk_x
                mstore(add(_pPairing, 384), mload(add(pMem, pVk)))
                mstore(add(_pPairing, 416), mload(add(pMem, add(pVk, 32))))


                // gamma2
                mstore(add(_pPairing, 448), gammax1)
                mstore(add(_pPairing, 480), gammax2)
                mstore(add(_pPairing, 512), gammay1)
                mstore(add(_pPairing, 544), gammay2)

                // C
                mstore(add(_pPairing, 576), calldataload(pC))
                mstore(add(_pPairing, 608), calldataload(add(pC, 32)))

                // delta2
                mstore(add(_pPairing, 640), deltax1)
                mstore(add(_pPairing, 672), deltax2)
                mstore(add(_pPairing, 704), deltay1)
                mstore(add(_pPairing, 736), deltay2)


                let success := staticcall(sub(gas(), 2000), 8, _pPairing, 768, _pPairing, 0x20)

                isOk := and(success, mload(_pPairing))
            }

            let pMem := mload(0x40)
            mstore(0x40, add(pMem, pLastMem))

            // Validate that all evaluations ∈ F
            
            checkField(calldataload(add(_pubSignals, 0)))
            
            checkField(calldataload(add(_pubSignals, 32)))
            
            checkField(calldataload(add(_pubSignals, 64)))
            
            checkField(calldataload(add(_pubSignals, 96)))
            
            checkField(calldataload(add(_pubSignals, 128)))
            
            checkField(calldataload(add(_pubSignals, 160)))
            
            checkField(calldataload(add(_pubSignals, 192)))
            
            checkField(calldataload(add(_pubSignals, 224)))
            
            checkField(calldataload(add(_pubSignals, 256)))
            
            checkField(calldataload(add(_pubSignals, 288)))
            
            checkField(calldataload(add(_pubSignals, 320)))
            
            checkField(calldataload(add(_pubSignals, 352)))
            
            checkField(calldataload(add(_pubSignals, 384)))
            
            checkField(calldataload(add(_pubSignals, 416)))
            

            // Validate all evaluations
            let isValid := checkPairing(_pA, _pB, _pC, _pubSignals, pMem)

            mstore(0, isValid)
            
            return(0, 0x20)
        }
    }
}